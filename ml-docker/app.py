from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)

# Load the Titanic dataset and prepare it
data = pd.read_csv("titanic.csv")
data = data[['Fare', 'Survived']].dropna()

# Route for data analysis and graph
@app.route('/analyze_data', methods=['GET'])
def analyze_data():
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Survived', y='Fare', data=data)
    plt.title("Ticket Fare vs. Survival")
    plt.savefig('graph.png')  # Save the graph as an image file
    return "Data analysis and graph plotted."

# Route for model training
@app.route('/train_model', methods=['GET'])
def train_model():
    # Split the data into features (X) and target (y)
    X = data[['Fare']]
    y = data['Survived']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the Random Forest Classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    # Render the results template and pass the results as variables
    return render_template('results.html', accuracy=accuracy, classification_report=classification_rep)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
