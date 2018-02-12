import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.model_selection import 


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Flatten, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D
from keras import regularizers
import keras

#Import data
train_with_labels = pd.read_csv('training_data.txt',sep=' ')
y = train_with_labels['Label']
train = train_with_labels.drop('Label',1)
test = pd.read_csv('test_data.txt',sep=' ')
joint_set = pd.concat([train,test])

#Further data preprocessing
words = [str(k) for k in joint_set.keys()]
#Count of word appearence. 
#Consider pruning words that are too ubiquitous (and seem to be neutral) or not used enough

word_count = []

for word in words:
   word_count.append((np.sum(train[word]),word))
   #word_count.append((np.sum(test[word]),word))
   #word_count.append((np.sum(joint_set[word]),word))
   
#sorted(word_count)

#Individual word sentiment
y = y.as_matrix()

def get_sentiment(word, weighted = False):
   '''
   Input: word
   Output: average sentiment (1 for 4/5 star, 0 for 1/2 star of that word)
   If weighted == True: it counts each instance of word as a separate review for the purpose of average sentiment
   '''
   
   labels = np.array(train_with_labels.loc[train_with_labels[word]>0]['Label'])
   
   
   return round(np.mean(labels),3)




word_sentiment = []

for word in words:
   sentiment = get_sentiment(word)
   word_sentiment.append((np.sum(train[word]),sentiment,word))
   
#sorted(word_sentiment)
word_sentiment.sort(key=lambda x: x[1])

sentiment_table = pd.DataFrame(word_sentiment)
sentiment_table = sentiment_table.set_index(2)
# X_train,X_test,y_train,y_test = [np.array(df) for df in train_test_split(train,y,test_size=0.25,random_state = 47)]
#Try a CNN...

sorted_train = np.array(train[np.array(sentiment_table.index)])
sorted_test = np.array(test[np.array(sentiment_table.index)])

# X_train,X_test,y_train,y_test = [np.array(df) for df in train_test_split(sorted_train,y,test_size=0.25,random_state = 47)]


NSPLITS = 4 #number of k-fold validations
N_NETS = 3  #number of networks generated

kf = KFold(n_splits = NSPLITS, shuffle = False)
accuracy = np.zeros(2)

predictions = np.zeros((20000,2))
bi = 0
for (train_idx, test_idx) in kf.split(sorted_train):
	tot_acc = 0

	
	
	X_train, y_train = sorted_train[train_idx], y[train_idx]
	y_test, X_test = y[test_idx], sorted_train[test_idx]
	# we must reshape the X data (add a channel dimension)
	X_train = X_train.reshape(tuple(list(X_train.shape)+[1]))
	X_test = X_test.reshape(tuple(list(X_test.shape)+[1]))
	# we'll need to one-hot encode the labels
	y_train = keras.utils.np_utils.to_categorical(y_train)
	y_test = keras.utils.np_utils.to_categorical(y_test)

	len_test_set = len(y_test)

	ksize = 5
	mpool = 2

	for i in range(N_NETS):
		model = Sequential()
		model.add(Conv1D(32, kernel_size=ksize, padding='same',input_shape=(1000,1)))
		model.add(Activation('relu'))
		model.add(MaxPooling1D(pool_size=mpool))
		model.add(Dropout(0.1))

		model.add(Conv1D(8, kernel_size=ksize, padding='same'))
		model.add(Conv1D(8, kernel_size=ksize, padding='same'))
		model.add(Activation('relu'))
		model.add(MaxPooling1D(pool_size=mpool))
		model.add(Dropout(0.1))

		model.add(Flatten())
		model.add(Dense(100))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(Dropout(0.5))
		model.add(Dense(2))
		model.add(Activation('softmax'))

		print ('Number of parameters = ', model.count_params())

		model.compile(optimizer='adagrad',loss='categorical_crossentropy',metrics=['accuracy'])
		history = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))
		train_loss, train_acc = model.evaluate(x=X_train, y=y_train)
		test_loss, test_acc = model.evaluate(x=X_test, y=y_test)

		print ("train, test acc", train_acc, test_acc)
		accuracy += (train_acc,test_acc)
		# print(model.predict(X_test))
		predicted = model.predict(X_test)
		for i,j in enumerate(test_idx):
			# print(test_idx)
			# print(i,j)
			# print(predicted)
			# print(predicted[0])
			# print(predictions[0])
			predictions[j] += predicted[i]*test_acc
		tot_acc += test_acc

	for i,j in enumerate(test_idx):
		predictions[j] /= tot_acc

	print("Intermediary RESULT:")
	print("prediction is ")
	p = np.zeros(len_test_set)
	for i,j in enumerate(test_idx):
		p[i] = (1 if predictions[j][1]>.5 else 0)
		p[i] = p[i] == y[j]
	print(np.sum(p)/len_test_set)
	print("% accurate \n")
	bi += 1

accuracy /= NSPLITS*N_NETS
print('Test score:', accuracy[0])
print('Test accuracy:', accuracy[1])


print("FINAL RESULT:")
print("prediction is ")
# p = np.zeros(len(y_test))
for i in range(len(y_test)):
	p[i] = (1 if predictions[i][1]>.5 else 0)
	p[i] = p[i] == y[i]
print(np.sum(p)/len(y_test))
print("% accurate \n")

np.save("predict.npy", predictions)
print()


