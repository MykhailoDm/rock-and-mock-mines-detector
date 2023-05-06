## Getting Started

1\. Clone Git repository:

If needed, install `git`.

Run:

```
git clone https://github.com/MykhailoDm/rock-and-mock-mines-detector.git
```

## <a name="gettingstarted1"></a>Running Code for trained model

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
python -m flask run
```
*Note:* Location of a model is specified in a MODEL_LOCATION at the start of the file 