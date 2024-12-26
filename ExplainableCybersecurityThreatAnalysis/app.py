import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import shap
from flask import Flask, request, render_template, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load dataset
DATASET_PATH = "phishing_email.csv"
data = pd.read_csv(DATASET_PATH)

# Preprocess dataset
data["Email Text"] = data["Email Text"].fillna("")  # Handle missing values
y = data["Email Type"].apply(lambda x: 1 if x == "Phishing" else 0)  # Convert labels to numeric
X_raw = data["Email Text"]  # Extract raw text

# Transform text using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(X_raw)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# SHAP Explainer
explainer = shap.TreeExplainer(model)

# Flask routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input email text from the form
        email_text = request.form.get("email_text", "")

        if not email_text.strip():
            return jsonify({
                "error": "Input email text cannot be empty."
            }), 400

        # Transform input using the TF-IDF vectorizer
        input_vectorized = vectorizer.transform([email_text])

        # Make prediction
        prediction = model.predict(input_vectorized)[0]

        # Generate SHAP values
        shap_values = explainer.shap_values(input_vectorized)
        print(f"SHAP values: {shap_values}")  # Debugging line to check shap_values
        shap_values = shap_values[1][0].tolist()  # Convert SHAP values for the phishing class

        return jsonify({
            "prediction": "Phishing" if prediction == 1 else "Safe",
            "shap_values": shap_values,
            "feature_names": vectorizer.get_feature_names_out().tolist()
        })
    except Exception as e:
        print(f"Error during prediction: {str(e)}")  # Log the error in the server
        return jsonify({
            "error": "An error occurred during prediction.",
            "details": str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True)
