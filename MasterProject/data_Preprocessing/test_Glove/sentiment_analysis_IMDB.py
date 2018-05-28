import numpy as np
from keras.datasets import imdb
from matplotlib import pyplot
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from sklearn.model_selection import StratifiedKFold
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

#load the dataset

vocab = 5000
max_length = 500
seed = 7
np.random.seed(seed)

(x_train, y_train),(x_test,y_test) = imdb.load_data(num_words=vocab)


x_train = sequence.pad_sequences(x_train,maxlen=max_length)
x_test = sequence.pad_sequences(x_test,maxlen=max_length)




#define 10 fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
acc_scores = []

# MLP
# for train, validation in kfold.split(x_train,y_train):
# #create the model
#     model = Sequential()
#     model.add(Embedding(vocab, 32, input_length=500))
#     model.add(Flatten())
#     model.add(Dense(250,activation='relu'))
#     model.add(Dense(1,activation='sigmoid'))
#     model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#
#
#     model.fit(x_train[train],y_train[train],epochs=2,batch_size=128,verbose=0)
#
#     #evaluation
#     scores = model.evaluate(x_train[validation],y_train[validation],verbose=0)
#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
#     acc_scores.append(scores[1] * 100)
#
# print("%.2f%% (+/- %.2f%%)" % (np.mean(acc_scores), np.std(acc_scores)))

for train, validation in kfold.split(x_train,y_train):

    #define CNN neural network
    model = Sequential()
    model.add(Embedding(vocab, 32, input_length=max_length))
    model.add(Conv1D(filters=32,kernel_size=3, padding='same',activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add((Dense(250,activation='relu')))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    #fit the model
    model.fit(x_train[train],y_train[train],epochs=2,batch_size=128,verbose=0)

    #evaluation
    scores = model.evaluate(x_train[validation],y_train[validation],batch_size=128, verbose=0)

    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    acc_scores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(acc_scores), np.std(acc_scores)))
