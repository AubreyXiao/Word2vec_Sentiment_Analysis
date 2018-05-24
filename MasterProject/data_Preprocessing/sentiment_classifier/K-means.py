import time
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
import numpy as np
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


#create bag of centroids
def create_centroids(words, word_centroid_map):
    num_centroids = max(word_centroid_map.values())+1

    centroids_matrix = np.zeros(num_centroids, dtype="float32")

    for word in words:
        if(word in word_centroid_map):
            index = word_centroid_map[word]
            centroids_matrix[index] += 1
    return centroids_matrix


#get all the review
def get_clean_review_lists(review_datasets):
    clean_review_datasets = []
    for review in review_datasets:
        clean_review_datasets.append(clean_text(review))
    return clean_review_datasets


#preprocess the text and split each review to a list of tokens
def clean_text(review):
    #1: remove the tags
    review = BeautifulSoup(review).get_text()
    #2.remove the non alpha
    text = re.sub("[^a-zA-Z]", " ",review)
    #3.remove split the tokens
    lowercase =  text.lower().split()
    #remove the stopwords
    stops = set(stopwords.words("English"))
    words = [w for w in lowercase if not w in stops]

    return words


#----load the model-------
modelname = "/Users/xiaoyiwen/Desktop/Sentiment_Analysis_word2vec/MasterProject/data_Preprocessing/300features_40minwords_10window"
model = Word2Vec.load(modelname)

#-------------start time ------
start = time.time()

#-----set 5words per cluster----
wordvectors = model.wv.syn0
print(wordvectors)
num_clusters = wordvectors.shape[0]/5
print(num_clusters)

num_clusters = int(num_clusters)
print(num_clusters)
#initialize a k-means
kmean_cluster = KMeans(n_clusters=num_clusters)
idx = kmean_cluster.fit_predict(wordvectors)

end = time.time()
elapsed = end - start

print("Time taken for Kmeans clustering:", elapsed, "seconds.")

#create a word /index dict
word_centroid_map = dict(zip(model.wv.index2word, idx))

for cluster in range(0,10):
    print("\nCluster %d" % cluster)
    words=[]
    for key, item in word_centroid_map.items():
        if(item == cluster):
            words.append(key)
    print(words)




#_---------------------get reviews------------------

#---load the datasetets-----
dir = "/Users/xiaoyiwen/Desktop/Sentiment_Analysis_word2vec/MasterProject/data_Preprocessing/sentiment_classifier/kaggle_dataset/"

train_data = pd.read_csv(dir +"labeledTrainData.tsv",header = 0, delimiter='\t', quoting=3)
unlabeled_data = pd.read_csv(dir + "unlabeledTrainData.tsv",header = 0, delimiter='\t',quoting=3)
test_data = pd.read_csv(dir + "testData.tsv",header=0, delimiter='\t',quoting=3)

#-------get clean datasets--------------------------
cleaned_train_datasets = get_clean_review_lists(train_data["review"])
cleaned_test_datastes = get_clean_review_lists(test_data["review"])


#-------create train bags of centroids----------------
train_count = 0
train_matrix = np.zeros((train_data["review"].size, num_clusters), dtype="float32")

for word_lists in cleaned_train_datasets:

    train_matrix[train_count] =  create_centroids(word_lists, word_centroid_map)
    train_count += 1


#-------create test bags of centroids----------------

test_count = 0
test_matrix = np.zeros((test_data["review"].size, num_clusters),dtype="float32")

for words in cleaned_test_datastes:
    test_matrix[test_count] = create_centroids(words, word_centroid_map)
    test_count += 1


#------------fit in the randdom forest tree--------------

forest = RandomForestClassifier(n_estimators=100)

#fit the forest
forest = forest.fit(train_matrix, train_data["sentiment"])
result = forest.predict(test_matrix)

#write the output
output = pd.DataFrame(data={"id": test_data["id"], "sentiment":result})
output.to_csv("k-means_clustering.csv",index=False,quoting=3)






























