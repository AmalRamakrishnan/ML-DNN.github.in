from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import numpy
 
# Function to create model, required for KerasClassifier
def create_model(optimizer='rmsprop', init='glorot_uniform', dropout_rate=0.0):
    # create model
    #we designed a DNN model with two hidden layers. 
    #The first hidden layer constituted the number of 
    #neurons amounting to 50% of attributes of the 
    #input feature space. Subsequently, the second layer 
    #contained 50% of the neurons that were present in the previous layer.
    #For example, if the feature set contains 1000 attributes, 
    #then the first layer will be created with 500 neurons and 
    #the second layer is formed using 250 neurons.
    model = Sequential()
    model.add(Dense(4, input_dim=8, kernel_initializer=init, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(2, kernel_initializer=init, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
 
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians diabetes dataset
dataset = numpy.loadtxt("pima-indians-diabetes.data.csv", delimiter=",") 
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
y = dataset[:,8]
#Data Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
RescaledX = sc.fit_transform(X)
#Spliting Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(RescaledX, y, test_size =0.4, random_state=4)
# create model
model = KerasClassifier(build_fn=create_model, verbose=0)
#Parameters
# grid search init, epochs, batch size,optimizer and dropout rate
dropout_rate = [0.0]

optimizers = ['rmsprop', 'adam']
init = ['glorot_uniform', 'normal', 'uniform']
epochs = [50, 100, 150]
batches = [2,3]


param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init, dropout_rate=dropout_rate)
rcv = RandomizedSearchCV(estimator=model, param_distributions=param_grid)
rcv_result = rcv.fit(X_train, y_train)
#rcv.best_estimator_.fit(X_train, y_train)
y_pred = rcv.best_estimator_.predict(X_test)
print('Accuracy      :',accuracy_score(y_test, y_pred), rcv_result.best_params_)
print('Precision     :',precision_score(y_test,y_pred))
print('Recall        :',recall_score(y_test,y_pred))
print('F1_Score      :',f1_score(y_test,y_pred))