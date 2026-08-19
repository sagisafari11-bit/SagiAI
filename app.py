
from flask import Flask, render_template, request, jsonify
import requests
import json
import re

from prediction import predict_contact


app = Flask(__name__)


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"


PATTERNS = [
    {
        "name": "Abandonment Sensitivity",
        "description": (
            "Fear, distress, or strong concern about losing a partner, "
            "being left, being rejected, or the relationship ending."
        )
    },
    {
        "name": "Emotional Avoidance",
        "description": (
            "Withdrawing, becoming silent, disappearing, avoiding difficult "
            "conversations, or pulling away emotionally during conflict."
        )
    },
    {
        "name": "Emotional Inconsistency",
        "description": (
            "Alternating between closeness and distance, affection and "
            "coldness, or otherwise inconsistent emotional behaviour."
        )
    },
    {
        "name": "Reassurance Seeking",
        "description": (
            "Repeatedly seeking confirmation, reassurance, validation, "
            "or certainty about a partner's feelings or commitment."
        )
    },
    {
        "name": "Relationship Instability",
        "description": (
            "Repeated breakups, reconciliations, on-and-off relationships, "
            "or repeated cycles of separation and reunion."
        )
    },
    {
        "name": "Social Media Preoccupation",
        "description": (
            "Repeatedly checking, monitoring, watching stories, viewing "
            "profiles, or focusing heavily on a partner's social-media activity."
        )
    }
]


# =========================================================
# FEATURES REQUIRED BY THE SKLEARN MODEL
# =========================================================

REQUIRED_FEATURES = [
    "communication_frequency",
    "usual_response_time_hours",
    "conflict_withdrawal",
    "previous_return",
    "initiates_contact",
    "social_media_monitoring",
    "relationship_length_months",
    "previous_breakups",
    "days_without_contact"
]


# =========================================================
# QUESTIONS FOR INFORMATION NOT PROVIDED IN THE STORY
# =========================================================

FEATURE_QUESTIONS = {

    "communication_frequency":
        "How often did you usually communicate with each other?",

    "usual_response_time_hours":
        "When they were available, how quickly did they usually reply?",

    "conflict_withdrawal":
        "During arguments or conflict, how did they usually respond?",

    "previous_return":
        "After previous periods of separation, did they ever come back or reconnect?",

    "initiates_contact":
        "After periods of no contact, who usually reached out first — you or them?",

    "social_media_monitoring":
        "After the breakup or during no contact, did they continue checking or viewing your social media?",

    "relationship_length_months":
        "Approximately how long were you together?",

    "previous_breakups":
        "How many times had you broken up before the most recent breakup?",

    "days_without_contact":
        "Approximately how many days have you gone without contact since the most recent interaction?"
}



def normalise(text):

    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_json(content):

    content = content.strip()

    if content.startswith("```"):

        content = re.sub(
            r"^```(?:json)?",
            "",
            content
        )

        content = re.sub(
            r"```$",
            "",
            content
        )

        content = content.strip()

    return json.loads(content)


def ask_ollama(story):

    pattern_text = "\n".join(
        f"{i + 1}. {p['name']}: {p['description']}"
        for i, p in enumerate(PATTERNS)
    )

    system_prompt = f"""
You are SagiAI, a behavioural-pattern analysis assistant.

Analyse the MEANING and CONTEXT of the user's relationship story.

DO NOT depend on exact keywords.

Understand behaviour even when the user describes it using
completely different wording.

For example:

"The second there was an argument, he would vanish for four days."

may indicate Emotional Avoidance.

"I was constantly terrified that eventually he would leave."

may indicate Abandonment Sensitivity.

The user does NOT need to use the names of the patterns.

The patterns you should examine are:

{pattern_text}

IMPORTANT:

Do not diagnose anyone.

Do not state psychological conditions as facts.

Use cautious language such as:

"may be consistent with..."
"could suggest..."
"may indicate..."

For EVERY pattern return:

- name
- score
- probability
- evidence
- explanation
- possible_interpretations

SCORING:

0 = no meaningful evidence
1 = weak or isolated evidence
2 = noticeable evidence
3 = strong or repeated evidence

PROBABILITY:

Return a number from 0 to 100.

This is NOT a medical or psychological probability.

It represents how strongly the story supports the behavioural pattern.

Suggested relationship:

score 0 -> approximately 0-10
score 1 -> approximately 25-45
score 2 -> approximately 50-75
score 3 -> approximately 75-100

IMPORTANT:

Evidence must come from the story.

Do NOT invent events.

If there is no evidence for a pattern:

score = 0
probability = 0
evidence = "No clear evidence in the story."
explanation = "There is not enough information to identify this pattern."
possible_interpretations = []

Return ONLY valid JSON.

Use exactly this structure:

{{
    "patterns": [
        {{
            "name": "Pattern name",
            "score": 0,
            "probability": 0,
            "evidence": "Evidence from the story.",
            "explanation": "Explanation.",
            "possible_interpretations": []
        }}
    ],
    "overall_observations": []
}}
"""

    user_prompt = f"""
Analyse this relationship story:

{story}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1
            }
        },
        timeout=180
    )

    response.raise_for_status()

    data = response.json()

    return clean_json(
        data["message"]["content"]
    )


def extract_features(story):

    prompt = f"""
You are SagiAI's information extraction system.

Read the relationship story and understand its meaning.

DO NOT use keyword matching.

DO NOT rely on exact phrases.

Infer information only when the story clearly provides it.

Do NOT guess.

If information is unavailable, return null.

Extract these nine features:

1. communication_frequency

1 = frequent communication
0 = infrequent communication
null = unknown


2. usual_response_time_hours

Approximate usual response time in hours.

"within two hours" = 2
"almost immediately" = 1
null = unknown


3. conflict_withdrawal

1 = they withdraw, disappear, become silent,
stop responding, avoid conflict, or pull away

0 = they generally communicate during conflict

null = unknown


4. previous_return

1 = they returned or reconnected after separation

0 = the story clearly indicates they did not return

null = unknown


5. initiates_contact

1 = the other person usually initiates contact after separation/no contact

0 = the user usually initiates contact

null = unknown


6. social_media_monitoring

1 = the other person monitors/checks the user's social media

0 = no indication

null = unknown


7. relationship_length_months

Convert years/months into months.

"one year" = 12
"two years" = 24
"18 months" = 18

null = unknown


8. previous_breakups

Number of breakups BEFORE the latest/current breakup.

Example:

"We broke up twice before the final breakup."

previous_breakups = 2

null = unknown


9. days_without_contact

Number of days without contact in the latest relevant period.

Example:

"We haven't spoken for six weeks."

days_without_contact = approximately 42

null = unknown


IMPORTANT:

Do not confuse the user's actions with the other person's actions.

For example:

"I contacted him after the breakup."

does NOT mean the other person initiated contact.

Return ONLY valid JSON.

Use exactly this structure:

{{
    "communication_frequency": null,
    "usual_response_time_hours": null,
    "conflict_withdrawal": null,
    "previous_return": null,
    "initiates_contact": null,
    "social_media_monitoring": null,
    "relationship_length_months": null,
    "previous_breakups": null,
    "days_without_contact": null
}}

STORY:

{story}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0
            }
        },
        timeout=180
    )

    response.raise_for_status()

    data = response.json()

    return clean_json(
        data["message"]["content"]
    )



def get_missing_features(features):

    missing = []

    for feature in REQUIRED_FEATURES:

        # 0 is a valid answer.
        # Only None/missing means we need to ask.
        if features.get(feature) is None:
            missing.append(feature)

    return missing


def interpret_answers(answers):

    if not answers:
        return {}

    answer_text = ""

    for key, value in answers.items():

        answer_text += f"""
FEATURE:
{key}

USER ANSWER:
{value}

"""

    prompt = f"""
You are SagiAI's structured information extraction system.

Interpret the user's answers by meaning.

Do NOT rely on exact keywords.

Do NOT guess.

Return null when an answer does not provide the requested information.

The answers are:

{answer_text}

Return ONLY valid JSON.

Use this structure:

{{
    "communication_frequency": null,
    "usual_response_time_hours": null,
    "conflict_withdrawal": null,
    "previous_return": null,
    "initiates_contact": null,
    "social_media_monitoring": null,
    "relationship_length_months": null,
    "previous_breakups": null,
    "days_without_contact": null
}}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0
            }
        },
        timeout=180
    )

    response.raise_for_status()

    data = response.json()

    return clean_json(
        data["message"]["content"]
    )


def run_prediction(features):

    prediction_result = predict_contact(

        communication_frequency=features[
            "communication_frequency"
        ],

        usual_response_time_hours=features[
            "usual_response_time_hours"
        ],

        conflict_withdrawal=features[
            "conflict_withdrawal"
        ],

        previous_return=features[
            "previous_return"
        ],

        initiates_contact=features[
            "initiates_contact"
        ],

        social_media_monitoring=features[
            "social_media_monitoring"
        ],

        relationship_length_months=features[
            "relationship_length_months"
        ],

        previous_breakups=features[
            "previous_breakups"
        ],

        days_without_contact=features[
            "days_without_contact"
        ]
    )

    # Convert sklearn probability from:
    # 0.0 - 1.0
    #
    # into:
    # 0 - 100
    #
    # while preserving the original value too.

    probability = prediction_result.get(
        "probability",
        0
    )

    prediction_result["probability_percent"] = round(
        float(probability) * 100,
        1
    )

    # Human-readable result

    if prediction_result.get("prediction") == 1:

        prediction_result["result"] = (
            "More likely to contact again"
        )

    else:

        prediction_result["result"] = (
            "Less likely to contact again"
        )

    return prediction_result


@app.route("/")
def home():

    return render_template("index.html")


@app.route("/advice", methods=["POST"])
def advice():

    data = request.get_json(silent=True) or {}

    story = normalise(
        data.get("story", "")
    )

    if not story:

        return jsonify({
            "error": "Please tell me what happened."
        }), 400

    try:


        schema_analysis = ask_ollama(
            story
        )


        features = extract_features(
            story
        )

      

        missing = get_missing_features(
            features
        )


        if missing:

            questions = []

            for feature in missing:

                questions.append({

                    "feature": feature,

                    "question":
                        FEATURE_QUESTIONS[feature]

                })

            return jsonify({

                "status": "questions",

                "schema_analysis":
                    schema_analysis,

                "features":
                    features,

                "questions":
                    questions

            })


      

        prediction = run_prediction(
            features
        )

        return jsonify({

            "status": "complete",

            "schema_analysis":
                schema_analysis,

            "features":
                features,

            "prediction":
                prediction,

            "questions": []

        })

    except requests.exceptions.ConnectionError:

        return jsonify({

            "error":
                "SagiAI could not connect to Ollama. "
                "Make sure Ollama is running."

        }), 503

    except requests.exceptions.Timeout:

        return jsonify({

            "error":
                "Ollama took too long to respond. "
                "Please try again."

        }), 504

    except Exception as e:

        print(
            "ERROR IN /advice:",
            repr(e)
        )

        return jsonify({

            "error":
                str(e)

        }), 500



@app.route("/answer", methods=["POST"])
def answer():

    data = request.get_json(silent=True) or {}

    story = normalise(
        data.get("story", "")
    )

    if not story:

        return jsonify({
            "error": "The story is missing."
        }), 400

    answers = data.get(
        "answers",
        {}
    )

    try:


        story_features = extract_features(
            story
        )

       

        answer_features = interpret_answers(
            answers
        )

        # ---------------------------------------------
        # 3. MERGE
        # ---------------------------------------------

        combined_features = dict(
            story_features
        )

        for key, value in answer_features.items():

            if value is not None:

                combined_features[key] = value

       

        missing = get_missing_features(
            combined_features
        )

    

        if missing:

            questions = []

            for feature in missing:

                questions.append({

                    "feature": feature,

                    "question":
                        FEATURE_QUESTIONS[feature]

                })

            return jsonify({

                "status": "questions",

                "features":
                    combined_features,

                "questions":
                    questions

            })


        schema_analysis = ask_ollama(
            story
        )


        prediction = run_prediction(
            combined_features
        )

        return jsonify({

            "status": "complete",

            "schema_analysis":
                schema_analysis,

            "features":
                combined_features,

            "prediction":
                prediction,

            "questions": []

        })

    except requests.exceptions.ConnectionError:

        return jsonify({

            "error":
                "SagiAI could not connect to Ollama. "
                "Make sure Ollama is running."

        }), 503

    except requests.exceptions.Timeout:

        return jsonify({

            "error":
                "Ollama took too long to respond."

        }), 504

    except Exception as e:

        print(
            "ERROR IN /answer:",
            repr(e)
        )

        return jsonify({

            "error":
                str(e)

        }), 500


if __name__ == "__main__":

    print()
    print("======================================")
    print("          SagiAI STARTING")
    print("======================================")
    print()
    print("Ollama model:", MODEL)
    print()
    print("Website:")
    print("http://127.0.0.1:5000")
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )

