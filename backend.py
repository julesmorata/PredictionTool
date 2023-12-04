# app.py
import numpy as np
import tensorflow as tf
import tensorflow_text as text
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    model = tf.saved_model.load('mcode_bert2')
    text = [request.files['file'].read()]
    probs = np.array(model(text))
    result = {'Stage 1': str(probs[0][0]), 'Stage 2': str(probs[0][1]), 'Stage 3': str(probs[0][2]), 'Stage 4': str(probs[0][3])}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
