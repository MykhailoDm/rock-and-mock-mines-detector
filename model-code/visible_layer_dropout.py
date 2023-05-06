from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pandas import read_csv
from scikeras.wrappers import KerasClassifier

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
    # створюємо модель і задаємо леєри, Додається Dropout до Visible Layer
    mdl = Sequential()
    mdl.add(Dropout(0.5, input_shape=(60,)))
    mdl.add(Dense(60, input_shape=(60,), activation='relu'))
    mdl.add(Dense(30,  activation='relu'))
    mdl.add(Dense(1, activation='sigmoid'))
    # компілюємо модель
    sgd = SGD(learning_rate=0.01, momentum=0.8)
    mdl.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return mdl
estmtrs.append(('mlp', KerasClassifier(model=pplncreate_with_dropout, epochs=300, batch_size=16, verbose=0)))
# пайплайн
ppln = Pipeline(estmtrs)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(ppln, X, encoded_Y, cv=kfold)
print("Dropout: %.2f%% " % (results.mean()*100))
