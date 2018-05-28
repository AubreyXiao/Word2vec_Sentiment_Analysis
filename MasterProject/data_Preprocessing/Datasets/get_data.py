import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data
import numpy as np
from wordcloud import WordCloud
from matplotlib import pyplot as plt
import sys

class Datasets():


   def get_sent_detector(self):

       sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
       return sent_detector


   #get datasets from the csv--------
   #2：
   def get_train_data(self):

        train_data = pd.read_csv("/Users/xiaoyiwen/Desktop/Sentiment_Analysis_word2vec/MasterProject/data_Preprocessing/Datasets/kaggle_data/labeledTrainData.tsv",header = 0, delimiter='\t', quoting=3)
        return train_data

   #3：
   def get_test_data(self):

        test_data = pd.read_csv("/Users/xiaoyiwen/Desktop/Sentiment_Analysis_word2vec/MasterProject/data_Preprocessing/Datasets/kaggle_data/testData.tsv",header=0, delimiter='\t',quoting=3)
        return test_data

   #4：
   def get_unlabeled_data(self):


        unlabeled_data = pd.read_csv("/Users/xiaoyiwen/Desktop/Sentiment_Analysis_word2vec/MasterProject/data_Preprocessing/Datasets/kaggle_data/unlabeledTrainData.tsv", header=0, delimiter='\t', quoting=3)
        return unlabeled_data

   #5：
   # get all the review
   def get_clean_review_lists(self,review_datasets):
       clean_review_datasets = []
       for review in review_datasets:
           clean_review_datasets.append(Datasets.clean_text(review))
       return clean_review_datasets


   #clean the datasets------------
   #6：
   # preprocess the text and split each review to a list of tokens
   def clean_text(review):
       # 1: remove the tags
       review = BeautifulSoup(review).get_text()
       # 2.remove the non alpha
       text = re.sub("[^a-zA-Z]", " ", review)
       # 3.remove split the tokens
       lowercase = text.lower().split()
       # 4:remove the stopwords
       stops = set(stopwords.words("English"))
       words = [w for w in lowercase if not w in stops]

       return words


   #7：
   #split each review into sentences and clean the sentences -> to a list
   def transfer_review_to_sentences(review, sent_detector):

        #split the review to a list
        sentences = sent_detector.tokenize(review.strip())

        sentencs_list = []
        for sentence in sentences:
            if(len(sentence)>0):
             words = Datasets.clean_text(sentence)
             sentencs_list.append(words)

        return sentencs_list

   #8：
   def transfer_datasets_to_sentences(self,datasets,sent_detector):

       all_sentences = []

       for review in datasets:
           all_sentences += Datasets.transfer_review_to_sentences(review,sent_detector)

       return all_sentences

   #9：
   #get averaged vector review
   def return_averaged_vector_review(dimension, review, model):

       review_vector = np.zeros((dimension,), dtype="float32")

       vocab = set(model.wv.index2word)

       num_words = 0

       for word in review:
           if (word in vocab):
               num_words += 1
               review_vector = np.add(review_vector, model[word])

       # divide the result
       review_vector = np.divide(review_vector, num_words)

       return review_vector


   #10：
   def return_total_vector(self, reviews, model, dimension):

       # should be 25000(train reviews) * 300(dimension)
       matrix = np.zeros((len(reviews), dimension), dtype="float32")

       count = 0

       for review in reviews:

           if (count % 1000 == 0):
               print("Review %d of %d" % (count, len(reviews)))

           matrix[count] = Datasets.return_averaged_vector_review(dimension, review, model)
           count = count + 1

       return matrix


   #11：
   # plot word cloud
   def plot_word_cloud(self, terms):
       text = terms.index
       text = ' '.join(list(text))

       # lower the max fontsize
       wordcloud = WordCloud(max_font_size=40).generate(text)
       plt.figure(figsize=(25, 25))
       plt.imshow(wordcloud, interpolation="bilinear")
       plt.axis("off")
       plt.show()



