from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

# Load the trained model
model_filename = "titanic_model.pkl"
with open(model_filename, "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        pclass = int(request.form["Pclass"])
        sex = int(request.form["Sex"])  # Male=1, Female=0
        age = float(request.form["Age"])
        sibsp = int(request.form["SibSp"])
        parch = int(request.form["Parch"])
        fare = float(request.form["Fare"])
        embarked_C = int(request.form["Embarked_C"])
        embarked_Q = int(request.form["Embarked_Q"])

        # Preprocess input (match training data)
        input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked_C, embarked_Q]])

        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]  # Probability of survival

        result = "Survived" if prediction == 1 else "Did not survive"
        return render_template("index.html", prediction_text=f"Prediction: {result} ({probability:.2f} probability)")

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
