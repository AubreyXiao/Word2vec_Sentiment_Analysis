from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation




def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("TOpic %d" % (topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[: -no_top_words -1 : -1]]))

dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers','footers','quotes'))

#返回的是列表list
document = dataset.data
label = dataset.target

print(document)
#print (label)

no_features = 1000

tf_vectorizer= CountVectorizer(max_df=0.95,min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(document)
tf_feature_names = tf_vectorizer.get_feature_names()

num_topics = 20

#run LDA
lda = LatentDirichletAllocation(n_topics=num_topics, max_iter=5, learning_method='online',learning_offset=50., random_state=0).fit(tf)

no_top_words = 10

#display_topics(lda, tf_feature_names, no_top_words)