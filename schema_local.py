import json
import requests


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"


PATTERNS = {
    "Abandonment Sensitivity": (
        "Fear, strong concern, or difficulty coping with losing a partner, "
        "being left, or the relationship ending."
    ),

    "Emotional Avoidance": (
        "Withdrawing, becoming silent, disappearing, avoiding difficult "
        "conversations, or emotionally pulling away during conflict."
    ),

    "Emotional Inconsistency": (
        "Alternating between closeness and distance, affection and coldness, "
        "or otherwise inconsistent emotional behaviour."
    ),

    "Reassurance Seeking": (
        "Repeatedly seeking confirmation, reassurance, validation, or "
        "certainty about a partner's feelings or commitment."
    ),

    "Relationship Instability": (
        "Repeated breakups, reconciliations, on-and-off relationships, "
        "or repeated cycles of separation and reunion."
    ),

    "Social Media Preoccupation": (
        "Repeatedly checking, monitoring, watching stories, viewing profiles, "
        "or focusing heavily on a partner's social-media activity."
    )
}



def normalise(text):

    if not text:
        return ""

    return " ".join(text.split())


def call_ollama(system_prompt, user_prompt):

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
                "temperature": 0
            }
        },

        timeout=180
    )

    response.raise_for_status()

    data = response.json()

    content = data["message"]["content"]

    return json.loads(content)


def analyze_story(story):

    story = normalise(story)

    if not story:

        return {
            "patterns": [],
            "overall_observations": []
        }

    pattern_description = "\n".join(
        f"{name}: {description}"
        for name, description in PATTERNS.items()
    )

    system_prompt = f"""
You are SagiAI's behavioural analysis system.

Analyse the meaning and context of the user's relationship story.

Do NOT rely on exact keywords.

The user may describe a behaviour in many different ways.
You must understand the underlying meaning rather than search for
specific words.

The six predefined SagiAI behavioural patterns are:

{pattern_description}


SCORING

For every pattern give a score from 0 to 3.

0 = no meaningful evidence
1 = weak or isolated evidence
2 = noticeable evidence
3 = strong or repeated evidence


EVIDENCE

For every score, explain what in the story supports it.

Do not invent events that are not present in the story.


PSYCHOLOGICAL INTERPRETATIONS

You MAY mention possible psychological or behavioural interpretations.

However, they MUST be presented as possibilities.

For example:

"This may be consistent with emotional avoidance."

"This could suggest difficulty tolerating conflict."

"This may be consistent with attachment-related difficulties,
although the story alone cannot establish an attachment style."

Never present an interpretation as a diagnosis or established fact.

Do NOT say:

"He has an avoidant attachment style."

"She has attachment issues."

"He is emotionally unavailable."

Instead use cautious language such as:

"This behaviour could be consistent with emotional unavailability."


IMPORTANT:

Separate:

1. What the person actually described.
2. Your interpretation of what that behaviour might mean.


Return ONLY valid JSON.

Use exactly this structure:

{{
    "patterns": [
        {{
            "name": "Abandonment Sensitivity",
            "score": 0,
            "evidence": "",
            "explanation": "",
            "possible_interpretations": []
        }},
        {{
            "name": "Emotional Avoidance",
            "score": 0,
            "evidence": "",
            "explanation": "",
            "possible_interpretations": []
        }},
        {{
            "name": "Emotional Inconsistency",
            "score": 0,
            "evidence": "",
            "explanation": "",
            "possible_interpretations": []
        }},
        {{
            "name": "Reassurance Seeking",
            "score": 0,
            "evidence": "",
            "explanation": "",
            "possible_interpretations": []
        }},
        {{
            "name": "Relationship Instability",
            "score": 0,
            "evidence": "",
            "explanation": "",
            "possible_interpretations": []
        }},
        {{
            "name": "Social Media Preoccupation",
            "score": 0,
            "evidence": "",
            "explanation": "",
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

    return call_ollama(
        system_prompt,
        user_prompt
    )



def extract_features(story):

    story = normalise(story)

    if not story:

        return {}


    system_prompt = """
You are SagiAI's structured information extraction system.

Read the user's relationship story and determine the values of the
prediction features below.

IMPORTANT:

DO NOT use keyword matching.

DO NOT use a predefined list of phrases.

DO NOT rely on exact wording.

Understand the meaning, context and relationships described in the story.

The same fact can be expressed in many different ways.

For example:

"we spoke pretty much every day"

and

"we were constantly messaging each other"

both indicate frequent communication.

Similarly:

"whenever we argued he would disappear for a while"

indicates conflict withdrawal even though the words "conflict withdrawal"
are never used.

ONLY extract information that is actually supported by the story.

NEVER invent missing information.

If the story does not provide enough information, return null.


=========================================================
FEATURES
=========================================================

communication_frequency

Return:

1 = frequent or regular communication
0 = infrequent communication
null = insufficient information


usual_response_time_hours

Return the approximate usual response time in hours.

Examples:

"he usually replied within two hours" → 2

"she normally responded after half a day" → approximately 12

"he replied almost immediately" → approximately 1

null = insufficient information


conflict_withdrawal

Return:

1 = the person tends to withdraw, become silent, disappear,
    avoid discussion, stop responding, or pull away during conflict

0 = the story indicates that they generally communicate or engage
    during conflict

null = insufficient information


previous_return

Return:

1 = the other person previously returned or reconnected after
    a breakup/separation/no-contact period

0 = the story indicates that the other person did not previously return

null = insufficient information


initiates_contact

Return:

1 = the OTHER PERSON usually initiates contact after separation
    or periods of no contact

0 = THE USER usually initiates contact

null = insufficient information


social_media_monitoring

Return:

1 = the other person monitors, checks, watches or follows
    the user's social-media activity

0 = the story indicates that this does not happen

null = insufficient information


relationship_length_months

Return the approximate relationship length in months.

Examples:

"we dated for one year" → 12

"we were together for a year and a half" → 18

"we had been together for two years" → 24

null = insufficient information


previous_breakups

Return the number of breakups BEFORE the most recent/current breakup.

Example:

"We broke up twice before the final breakup."

→ 2

Do NOT count the current/final breakup as a previous breakup.

null = insufficient information


days_without_contact

Return the number of days since the relevant current/no-contact period.

Examples:

"We haven't spoken for 40 days."

→ 40

"I haven't heard from him in about three weeks."

→ approximately 21

null = insufficient information


=========================================================
IMPORTANT DISTINCTIONS
=========================================================

Previous behaviour and current behaviour must be separated.

Example:

"We broke up twice before and he came back both times."

Return:

previous_breakups = 2
previous_return = 1


Example:

"I contacted him after the breakup."

This does NOT mean:

initiates_contact = 1

It means the USER initiated contact.

Therefore:

initiates_contact = 0


Example:

"He contacted me after I stopped messaging him."

This indicates:

initiates_contact = 1


=========================================================
OUTPUT
=========================================================

Return ONLY valid JSON.

Use exactly this structure:

{
    "communication_frequency": null,
    "usual_response_time_hours": null,
    "conflict_withdrawal": null,
    "previous_return": null,
    "initiates_contact": null,
    "social_media_monitoring": null,
    "relationship_length_months": null,
    "previous_breakups": null,
    "days_without_contact": null
}
"""

    user_prompt = f"""
Extract the prediction features from this relationship story:

{story}
"""

    return call_ollama(
        system_prompt,
        user_prompt
    )


def get_missing_features(features):

    required = [
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

    missing = []

    for feature in required:

        if feature not in features:
            missing.append(feature)

        elif features[feature] is None:
            missing.append(feature)

    return missing



def extract_answer_features(answers):

    features = {}

    if not answers:
        return features

    if not isinstance(answers, dict):
        return features

    for key, value in answers.items():

        if value is None:
            continue

        value = str(value).strip()

        if not value:
            continue

        try:

            features[key] = int(value)

        except ValueError:

            try:
                features[key] = float(value)

            except ValueError:
                features[key] = value

    return features
