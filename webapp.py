import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import pickle

app = Flask(__name__)

# Load the saved Keras model and tokenizer
MODEL_PATH = 'model.h5'
model = load_model(MODEL_PATH)
# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    

# Set the maximum length of input sequences
MAXLEN = 200

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the text input from the user
    text = request.form['text']

    # Vectorize the input text
    X = tokenizer.texts_to_sequences([text])
    X = pad_sequences(X, padding='post', maxlen=MAXLEN)

    # Use the model to make a prediction
    y_pred = model.predict(X)[0, 0]
    if y_pred > 0.5:
        sentiment = 'Positive'
    else:
        sentiment = 'Negative'

    # Render the results template with the predicted sentiment
    return render_template('results.html', text=text, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)