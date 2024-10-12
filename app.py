import os
from flask import Flask, request, jsonify
import joblib

# Initialize Flask app
app = Flask(__name__)

model_dir = os.path.join(os.path.dirname(__file__), 'model')

# Load the model and vectorizer using relative paths
model_path = os.path.join(model_dir, 'random_forest_model_mew.pkl')
vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer_new.pkl')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# API Route for predicting sentiment
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')

    # Transform the input text using the saved vectorizer
    text_vector = vectorizer.transform([text])  # No preprocessing here

    # Predict the sentiment using the saved model
    prediction = model.predict(text_vector)

    # Return the prediction as JSON response
    return jsonify({'sentiment': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
