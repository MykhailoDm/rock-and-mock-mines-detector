from sklearn.preprocessing import LabelEncoder
from pandas import read_csv
from scikeras.wrappers import KerasClassifier
import keras
from flask import Flask, jsonify, request


MODEL_LOCATION = "C:\навчання\Курсові\НМ\code\model-code\model"
SONAR_CSV_LOCATION = "C:\навчання\Курсові\НМ\code\model-code\sonar.csv"

dataset = read_csv(SONAR_CSV_LOCATION, header=None).values
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
saved_model = keras.models.load_model(MODEL_LOCATION)
classifier = KerasClassifier(model=saved_model, epochs=300, batch_size=16, verbose=0)
classifier.initialize(X, encoded_Y)
# R - 1. M - 0.

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello"

@app.route("/api/v1/analyzier", methods = ['POST'])
def determine_rock_or_mock_mine():
    content = request.json
    content["signals"]
    result = classifier.predict([
        content["signals"]
    ])
    response = {
        "type": "MINE" if result[0] == 0 else "ROCK"
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)