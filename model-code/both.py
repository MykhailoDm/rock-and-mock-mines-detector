from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pandas import read_csv
from scikeras.wrappers import KerasClassifier
from constants.constants import MODEL_FILE_SAVE_LOCATION
import pickle
import keras

# завантажуємо датасет
dataset = read_csv("sonar.csv", header=None).values
# розділяємо в інпут (X) та аутпут (Y) змінні
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]
# енкодемо занчення в інтеджери
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
 
# естімейшини 
estmtrs = []
estmtrs.append(('standardize', StandardScaler()))
def pplncreate_with_dropout():
    # створюємо модель і задаємо леєри, Додається Dropout до Visible Layer, додаємо L2 регуляризатори
    mdl = Sequential()
    dropout_weight = 0.5
    l2_reg_factor = 0.0001
    mdl.add(Dropout(dropout_weight, input_shape=(60,)))
    mdl.add(Dense(60, input_shape=(60,), activation='relu', kernel_regularizer=regularizers.l2(l2_reg_factor)))
    mdl.add(Dense(30,  activation='relu', kernel_regularizer=regularizers.l2(l2_reg_factor)))
    mdl.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(l2_reg_factor)))
    # компілюємо модель
    sgd = SGD(learning_rate=0.01, momentum=0.8)
    mdl.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return mdl
classifier = KerasClassifier(model=pplncreate_with_dropout, epochs=300, batch_size=16, verbose=0)
estmtrs.append(('mlp', classifier))
# пайплайн
ppln = Pipeline(estmtrs)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(ppln, X, encoded_Y, cv=kfold)
print("Dropout and L2: %.2f%% " % (results.mean()*100))


classifier.fit(X, encoded_Y)
print("Finished fitting")
classifier.model_.save(MODEL_FILE_SAVE_LOCATION)
saved_model = keras.models.load_model(MODEL_FILE_SAVE_LOCATION)
classifier = KerasClassifier(model=saved_model, epochs=300, batch_size=16, verbose=0)
classifier.initialize(X, encoded_Y)
# R - 1. M - 0.
res = classifier.predict([
    [0.0209,0.0191,0.0411,0.0321,0.0698,0.1579,0.1438,0.1402,0.3048,0.3914,0.3504,0.3669,0.3943,0.3311,0.3331,0.3002,0.2324,0.1381,0.3450,0.4428,0.4890,0.3677,0.4379,0.4864,0.6207,0.7256,0.6624,0.7689,0.7981,0.8577,0.9273,0.7009,0.4851,0.3409,0.1406,0.1147,0.1433,0.1820,0.3605,0.5529,0.5988,0.5077,0.5512,0.5027,0.7034,0.5904,0.4069,0.2761,0.1584,0.0510,0.0054,0.0078,0.0201,0.0104,0.0039,0.0031,0.0062,0.0087,0.0070,0.0042],
    [0.0203,0.0121,0.0380,0.0128,0.0537,0.0874,0.1021,0.0852,0.1136,0.1747,0.2198,0.2721,0.2105,0.1727,0.2040,0.1786,0.1318,0.2260,0.2358,0.3107,0.3906,0.3631,0.4809,0.6531,0.7812,0.8395,0.9180,0.9769,0.8937,0.7022,0.6500,0.5069,0.3903,0.3009,0.1565,0.0985,0.2200,0.2243,0.2736,0.2152,0.2438,0.3154,0.2112,0.0991,0.0594,0.1940,0.1937,0.1082,0.0336,0.0177,0.0209,0.0134,0.0094,0.0047,0.0045,0.0042,0.0028,0.0036,0.0013,0.0016],
    [0.0392,0.0108,0.0267,0.0257,0.0410,0.0491,0.1053,0.1690,0.2105,0.2471,0.2680,0.3049,0.2863,0.2294,0.1165,0.2127,0.2062,0.2222,0.3241,0.4330,0.5071,0.5944,0.7078,0.7641,0.8878,0.9711,0.9880,0.9812,0.9464,0.8542,0.6457,0.3397,0.3828,0.3204,0.1331,0.0440,0.1234,0.2030,0.1652,0.1043,0.1066,0.2110,0.2417,0.1631,0.0769,0.0723,0.0912,0.0812,0.0496,0.0101,0.0089,0.0083,0.0080,0.0026,0.0079,0.0042,0.0071,0.0044,0.0022,0.0014],
    [0.0366,0.0421,0.0504,0.0250,0.0596,0.0252,0.0958,0.0991,0.1419,0.1847,0.2222,0.2648,0.2508,0.2291,0.1555,0.1863,0.2387,0.3345,0.5233,0.6684,0.7766,0.7928,0.7940,0.9129,0.9498,0.9835,1.0000,0.9471,0.8237,0.6252,0.4181,0.3209,0.2658,0.2196,0.1588,0.0561,0.0948,0.1700,0.1215,0.1282,0.0386,0.1329,0.2331,0.2468,0.1960,0.1985,0.1570,0.0921,0.0549,0.0194,0.0166,0.0132,0.0027,0.0022,0.0059,0.0016,0.0025,0.0017,0.0027,0.0027],
    [0.0260,0.0363,0.0136,0.0272,0.0214,0.0338,0.0655,0.1400,0.1843,0.2354,0.2720,0.2442,0.1665,0.0336,0.1302,0.1708,0.2177,0.3175,0.3714,0.4552,0.5700,0.7397,0.8062,0.8837,0.9432,1.0000,0.9375,0.7603,0.7123,0.8358,0.7622,0.4567,0.1715,0.1549,0.1641,0.1869,0.2655,0.1713,0.0959,0.0768,0.0847,0.2076,0.2505,0.1862,0.1439,0.1470,0.0991,0.0041,0.0154,0.0116,0.0181,0.0146,0.0129,0.0047,0.0039,0.0061,0.0040,0.0036,0.0061,0.0115]
])
print(res)