from MasterProject.data_Preprocessing.Datasets import get_data
import numpy as np
from bs4 import BeautifulSoup
import re
from collections import Counter
from sklearn.cross_validation import train_test_split
import tensorflow as tf

data = get_data.Datasets()
train_data = data.get_train_data()

def clean_text(review):
    # 1: remove the tags
    review = BeautifulSoup(review).get_text()
    # 2.remove the non alpha
    text = re.sub("[^a-zA-Z]", " ", review)
    # 3.remove split the tokens
    words = text.lower().split()

    return words


#get words tokens list

review_lists_words =[]
words = []
for review in train_data["review"]:
    review_lists_words.append(clean_text(review))
    words += clean_text(review)

#create the counter
counts = Counter(words)

#creta
min5_vocab = {}
i = 1
min5_vocab["UNKNOW_WORD"]=0

for word, freq in counts.items():
    if(freq>5):
        min5_vocab[word]=i
        i +=1


print(len(min5_vocab))








vocab = sorted(counts, reverse=True, key=counts.get)
print("vocab size")
print(len(vocab))
print(vocab)






word_to_int = {word:i for i,word in enumerate(vocab,1)}
#print(word_to_int)

review_integer= []
for review in review_lists_words:
    review_integer.append([word_to_int[word] for word in review if word in word_to_int.keys()])

review_integer = [review[:200] for review in review_integer if len(review)>0]
print(len(review_integer))
review_lens = Counter([len(x) for x in review_integer])
print("zero: {}".format(review_lens[0]))
print("max: {}".format(max(review_lens)))

max_length = 200
feature_array = np.zeros((len(review_integer),max_length),dtype="int")

for index, integer in enumerate(review_integer):
    feature_array[index, -len(integer):] = np.array(integer)[:max_length]

print(feature_array[:10,])
print(len(feature_array))


#label array
label_array = np.array(train_data["sentiment"])

seed = 2000
#separate the test, validation, test dataset (0.8) ( 0.5)
x_train, x_test, y_train, y_test = train_test_split(feature_array,label_array,test_size=0.2,random_state=seed)

x_val,x_test,y_val,y_test = train_test_split(x_test, y_test, test_size=0.5)
print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(x_train.shape),
      "\nValidation set: \t{}".format(x_val.shape),
      "\nTest set: \t\t{}".format(x_test.shape))
print("label set: \t\t{}".format(y_train.shape),
      "\nValidation label set: \t{}".format(y_val.shape),
      "\nTest label set: \t\t{}".format(y_test.shape))

#
# #set paramters
# lstm_size = 256
# lstm_layers = 2
# batch_size = 1000
# learning_rate = 0.01
#
# #
# n_words = len(vocab)+1
# tf.reset_default_graph()
#
# with tf.name_scope('inputs'):
#     inputs_ = tf.placeholder(tf.int32,[None,None],name="input")
#     labels = tf.placeholder(tf.int32,[None,None],name="labels")
#     keep_prob = tf.placeholder(tf.float32,name="keep_prob")
#
# embed_size = 300
#
# with tf.name_scope("Embeddings"):
#     embedding = tf.Variable(tf.random_uniform((n_words,embed_size),-1,1))
#     embed = tf.nn.embedding_lookup(embedding,inputs_)
