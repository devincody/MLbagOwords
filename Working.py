import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import BatchNormalization
from keras import regularizers

from sklearn.model_selection import KFold

train = pd.read_csv('training_data.txt',sep=' ')
y = train['Label']
train = train.drop('Label',1)
test = pd.read_csv('test_data.txt',sep=' ')
joint_set = pd.concat([train,test])
words = [str(k) for k in joint_set.keys()]



Xp = train.as_matrix()
test = test.as_matrix()
Y = y.as_matrix()

le = 1000

# # reviews = 1000
# # X = np.zeros((reviews,le*le))

# for i in range(le):
# 	for j in range(le):
# 		X[:,i*le+j] = Xp[:reviews, j] * Xp[:reviews,i]


# X = np.log(Xp+1)
# X = [X for (Xp<6)]
X = Xp

act = []
wid = []
drop = []
layers = []

for ac in ['relu','tanh', 'softmax', 'elu', 'selu', 'sigmoid', 'linear']:
	for wi in [20, 200, 1000, 3000]:
		for dro in [.3, .4, .5, .6, .7, .8, .9]:
			for layers in [1, 2, 3]:
				NSPLITS = 4
				kf = KFold(n_splits = NSPLITS, shuffle = True)
				accuracy = np.zeros(2)
				for (train_idx, test_idx) in kf.split(X):
					X_train, Y_train = X[train_idx], Y[train_idx]
					Y_test, X_test = Y[test_idx], X[test_idx]

					# print( X_train2, Y_train2)
					#print (X_train, Y_train)

					#X_train = np.floor(X_train)
					# print(X_train)
					print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

					# X_train = np.log(X_train + 1)
					# X_test = np.log(X_test + 1)


					# ac = []
					# dp = []
					drop = .75
					reg = np.linspace(5e-5,1e-3,10)[1]
					## Create your own model here given the constraints in the problem
					model = Sequential()
					#model.add(BatchNormalization(axis=-1, input_shape=(1000,), momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
					model.add(Dense(wi, input_shape=(le,)))
					model.add(Activation(ac))
					model.add(Dropout(dro))

					for i in range(layers-1):
						model.add(Dense(wi,kernel_regularizer=regularizers.l1_l2(reg)))
						model.add(Activation(ac))
						model.add(Dropout(dro))

					# model.add(Dense(200))
					# model.add(Activation('relu'))
					# model.add(Dropout(drop))


					model.add(Dense(1))
					model.add(Activation('sigmoid'))

					## Printing a summary of the layers and weights in your model
					model.summary()
					# ad = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
					rp=keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0) # 84.9
																							# Adam 85.12	
																							#SG = 53.8		
																							#RMSprop default 85.2 /85.04
																							#Adagrad 85.26 / 85.18 /85.25
																							#adadelta 83.1
																							#adamax 84.78 / 84.82
																							#nadam 85.26 / 85.12
																							#TF optimizer 

					model.compile(loss='mse', optimizer='Adam', metrics=['accuracy'])

					#mse = 85.0

					fit = model.fit(X_train, Y_train, batch_size=500, epochs=10, verbose=1, validation_data =(X_test, Y_test))

					## Printing the accuracy of our model, according to the loss function specified in model.compile above
					score = model.evaluate(X_test, Y_test, verbose=0)
					print('Test score:', score[0])
					print('Test accuracy:', score[1])
					accuracy += score

				accuracy /= NSPLITS
				print('Test score:', accuracy[0])
				print('Test accuracy:', accuracy[1])






w1 = model.predict(test)
w2 = model.predict(Xp)
print(w1)
f = open("submission.txt","w")
f.write("Id,Prediction\n")

for i in range(len(test)):
	f.write("%d,%d\n"% (i,(1 if w1[i,0]>.5 else 0)))

f.close()
f = open("Validation_Predicitons.txt","w")	
f.write("Id,Prediction\n")
for i in range(len(Xp)):
	f.write("%d,%d\n"% (i,(1 if w2[i,0]>.5 else 0)))

f.close()
# 	dp.append(drop)
# 	ac.append(score[1])

# plt.plot(dp,ac)
# plt.show()





