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







train_with_labels = pd.read_csv('training_data.txt',sep=' ')
y = train_with_labels['Label']
train = train_with_labels.drop('Label',1)
test = pd.read_csv('test_data.txt',sep=' ')
joint_set = pd.concat([train, test])
words = [str(k) for k in joint_set.keys()]

# #Individual word sentiment
# def get_sentiment(word, weighted = False):
#    labels = np.array(train_with_labels.loc[train_with_labels[word]>0]['Label'])
#    return round(np.mean(labels),3)

# word_sentiment = []

# for word in words:
#    sentiment = get_sentiment(word)
#    word_sentiment.append((np.sum(train[word]),sentiment,word))
   
# #sorted(word_sentiment)
# word_sentiment.sort(key=lambda x: x[1])

# sentiments = np.array([tup[1] for tup in word_sentiment])
# freq = [tup[0] for tup in word_sentiment]
# ordered_words = [tup[2] for tup in word_sentiment]
# #Drop anything between 0.4 and 0.6 in sentiment
# neutral_words = np.array(ordered_words)[np.where((sentiments<0.6)&(sentiments>0.4))[0]]
# sparse_train = train.drop(neutral_words,axis=1)
# sparse_test = test.drop(neutral_words,axis=1)

# sparse_words = sparse_train.keys()



# Xp = sparse_train.as_matrix()
Y = y

le = 1000

#reviews = 5000
# X = np.zeros((reviews,le*le),dtype = np.int8)

# for i in range(le):
# 	for j in range(le):
# 		X[:,i*le+j] = Xp[:reviews, j] * Xp[:reviews,i]


# X = np.log(Xp+1)
# X = [X for (Xp<6)]
# X = Xp
X = train.as_matrix()

ac = np.zeros(6)
dp = []
# ([5, 10, 50], [0.83125, 0.8323499999999999, 0.83485])([407], [0.8304])
# ([0.7, 0.8, 0.9], [0.8242, 0.823, 0.7906]) #drop


NSPLITS = 4
kf = KFold(n_splits = NSPLITS, shuffle = True)
accuracy = np.zeros(2)
for (train_idx, test_idx) in kf.split(X):
	X_train, Y_train = X[train_idx], Y[train_idx]
	Y_test, X_test = Y[test_idx], X[test_idx]

	for n2 in [15000, 20000, 40000, 60000, 80000, 100000, 120000]:
		n_points = len(Y_train)
		print("Npoints", n_points)
		#n2 = 100000
		X_train2 = np.zeros((n2,1000))
		X_train2[:n_points] =X_train[:n_points]
		Y_train2 = np.zeros(n2)
		Y_train2[:n_points] = Y_train[:n_points]

		t = n_points
		while (t < n2):
			r = int(np.floor(np.random.uniform(0,n_points)))
			r2 = int(np.floor(np.random.uniform(0,n_points)))
			if(Y_train2[r] == Y_train2[r2]):
				X_train2[t] = (X_train2[r] + X_train2[r2])/2
				Y_train2[t] = Y_train2[r]*Y_train2[r2]
				t+=1


		# print( X_train2, Y_train2)
		print (X_train, Y_train)

		#X_train = np.floor(X_train)
		# print(X_train)

		print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

		# X_train = np.log(X_train + 1)
		# X_test = np.log(X_test + 1)

		drop = .75
		reg = np.linspace(5e-5,1e-3,10)[1]
		## Create your own model here given the constraints in the problem
		model = Sequential()
		#model.add(BatchNormalization(axis=-1, input_shape=(1000,), momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
		model.add(Dense(le, input_shape=(le,)))
		model.add(Activation('relu'))
		model.add(Dropout(drop))

		model.add(Dense(10,kernel_regularizer=regularizers.l1_l2(reg)))
		model.add(Activation('relu'))
		model.add(Dropout(drop))

		# model.add(Dense(200))
		# model.add(Activation('relu'))
		# model.add(Dropout(drop))


		model.add(Dense(1))
		model.add(Activation('sigmoid'))

		## Printing a summary of the layers and weights in your model
		model.summary()
		# ad = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
		rp=keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

		model.compile(loss='mse', optimizer=rp, metrics=['accuracy'])

		fit = model.fit(X_train2, Y_train2, batch_size=500, epochs=7, verbose=1, validation_data =(X_test, Y_test))

		## Printing the accuracy of our model, according to the loss function specified in model.compile above
		score = model.evaluate(X_test, Y_test, verbose=0)
		print('Test score:', score[0])
		print('Test accuracy:', score[1])
		# accuracy += score

		#accuracy /= NSPLITS
		print('Test score:', accuracy[0])
		print('Test accuracy:', accuracy[1])

		dp.append(n2)
		ac[int(n2/20000-1)] += score[1]
		print("so far:")
		print(dp, ac)


plt.plot(np.linspace(20,120,6),float(ac)/NSPLITS)
plt.show()





