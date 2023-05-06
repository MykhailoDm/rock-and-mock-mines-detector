## Getting Started

Clone Git repository:

If needed, install `git`.

Run:

```
git clone https://github.com/MykhailoDm/rock-and-mock-mines-detector.git
```

## <a name="gettingstarted1"></a>Running Code for trained model

sonar.csv dataset has been retrieved from http://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks).

For running scripts, which display accuracy in `print` statement, you can simply launch script like so:

```
python name_of_file.py
```

Code that is responsible for saving and loading model:

```
classifier.fit(X, encoded_Y)
classifier.model_.save(MODEL_FILE_SAVE_LOCATION)
saved_model = keras.models.load_model(MODEL_FILE_SAVE_LOCATION)
classifier = KerasClassifier(model=saved_model, epochs=300, batch_size=16, verbose=0)
classifier.initialize(X, encoded_Y)
# R - 1. M - 0.
res = classifier.predict([
    [0.0209,0.0191,0.0411,0.0321,0.0698,0.1579,0.1438,0.1402,0.3048,0.3914,0.3504,0.3669,0.3943,0.3311,0.3331,0.3002,0.2324,0.1381,0.3450,0.4428,0.4890,0.3677,0.4379,0.4864,0.6207,0.7256,0.6624,0.7689,0.7981,0.8577,0.9273,0.7009,0.4851,0.3409,0.1406,0.1147,0.1433,0.1820,0.3605,0.5529,0.5988,0.5077,0.5512,0.5027,0.7034,0.5904,0.4069,0.2761,0.1584,0.0510,0.0054,0.0078,0.0201,0.0104,0.0039,0.0031,0.0062,0.0087,0.0070,0.0042]
])
print(res)
```

Path for saved model is defined in model-code\constants\constants.py. The name of the model directory must be "model"

You may need to install dependencies with pip and requirements.txt, if you do not have required dependencies.

```
pip install -r requirements.txt
```

## Running Flask project.

For running flask project, you have to go to `web-app` directory. From there run:

```
python app.py
```
*Note:* Location of a model is specified in a MODEL_LOCATION constant at the start of the file. Location of sonar.csv is specified in a SONAR_CSV_LOCATION constant at the start of the file as well.

After application has started, you can visit the home page via going to http://127.0.0.1:5000/.
There you can provide sonar data, a comma-separated 60 numbers just like in sonar.csv. Example:


0.0164,0.0173,0.0347,0.0070,0.0187,0.0671,0.1056,0.0697,0.0962,0.0251,0.0801,0.1056,0.1266,0.0890,0.0198,0.1133,0.2826,0.3234,0.3238,0.4333,0.6068,0.7652,0.9203,0.9719,0.9207,0.7545,0.8289,0.8907,0.7309,0.6896,0.5829,0.4935,0.3101,0.0306,0.0244,0.1108,0.1594,0.1371,0.0696,0.0452,0.0620,0.1421,0.1597,0.1384,0.0372,0.0688,0.0867,0.0513,0.0092,0.0198,0.0118,0.0090,0.0223,0.0179,0.0084,0.0068,0.0032,0.0035,0.0056,0.0040


Then, after clicking submit, the page will be updated and you will see the response saying that type is either ROCK or MINE.
If you do not see response, try to correct data you send to server.