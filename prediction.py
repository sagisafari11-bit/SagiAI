import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# Load your dataset
data = pd.read_csv(
    r"C:\Users\Partofix.com\Desktop\New folder\relationship_cases.csv"
    
)


# The 9 features the model will learn from
features = [
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


# Input data
X = data[features]

# What we want to predict
y = data["contacted_again"]


# Create the Decision Tree
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)


# Train the model on your cases
model.fit(X, y)


# Function Flask will use
def predict_contact(
    communication_frequency,
    usual_response_time_hours,
    conflict_withdrawal,
    previous_return,
    initiates_contact,
    social_media_monitoring,
    relationship_length_months,
    previous_breakups,
    days_without_contact
):

    input_data = pd.DataFrame([{
        "communication_frequency": communication_frequency,
        "usual_response_time_hours": usual_response_time_hours,
        "conflict_withdrawal": conflict_withdrawal,
        "previous_return": previous_return,
        "initiates_contact": initiates_contact,
        "social_media_monitoring": social_media_monitoring,
        "relationship_length_months": relationship_length_months,
        "previous_breakups": previous_breakups,
        "days_without_contact": days_without_contact
    }])


    # Predict 0 or 1
    prediction = model.predict(input_data)[0]


    # Probability of each outcome
    probabilities = model.predict_proba(input_data)[0]


    return {
        "prediction": int(prediction),
        "probability": float(max(probabilities))
    }
