# SagiAI 

SagiAI is a machine learning application that analyses behavioural patterns in real-life relationship situations and predicts whether a person is likely to break a period of no contact.

The model is trained on a structured dataset based on real-life relationship scenarios. By analysing multiple behavioural and situational features, SagiAI identifies patterns associated with different outcomes and uses these patterns to generate a prediction for a new situation.

This project demonstrates the application of supervised machine learning to behavioural data, from data preparation and model training to prediction and integration into a web application.

## How It Works

SagiAI follows an end-to-end machine learning workflow:

Real-Life Relationship Scenarios  
↓  
Structured Dataset  
↓  
Feature Selection  
↓  
Data Preparation  
↓  
Model Training  
↓  
Decision Tree Classifier  
↓  
Behavioural Pattern Analysis  
↓  
New User Input  
↓  
Prediction

## Machine Learning Process

### 1. Dataset

SagiAI uses a structured dataset containing real-life relationship scenarios.

Each scenario is represented using multiple behavioural and situational features. The dataset provides examples that allow the machine learning model to learn patterns associated with whether someone breaks no contact.

### 2. Feature Selection

The relevant behavioural features are selected from each scenario and organised into a structured format that can be processed by the machine learning model.

Multiple features are considered together rather than relying on a single behavioural characteristic.

### 3. Model Training

SagiAI uses a Decision Tree Classifier implemented with Scikit-learn.

The model is trained on the relationship dataset and learns relationships between the input features and the target outcome.

A Decision Tree was selected because it can identify patterns between multiple features and produce an interpretable classification process.

### 4. Prediction

After the model has been trained, new behavioural and situational information can be provided to SagiAI.

The application processes the input and passes it through the trained machine learning model to generate a prediction of whether the person is likely to break no contact.

The prediction represents patterns learned from the available training data and is not a guaranteed prediction of future human behaviour.

## Machine Learning Workflow

1. Load the relationship dataset
2. Identify the relevant behavioural features
3. Prepare the data for machine learning
4. Define the target outcome
5. Train the Decision Tree classification model
6. Receive new user inputs
7. Process the inputs into the required format
8. Pass the inputs through the trained model
9. Generate the prediction
10. Display the prediction through the web application

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Decision Tree Classifier
- Flask
- HTML/CSS
- Git
- GitHub

## Project Structure
SagiAI/
│
├── app.py
├── prediction.py
├── schema_local.py
├── relationship_cases.csv
├── index.html
├── requirements.txt
├── README.md
└── .gitignore
