from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
#corpus
corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]

vectorizer = CountVectorizer()

x = vectorizer.fit_transform(corpus)
word = vectorizer.get_feature_names()
print(word)
print(x.toarray())


transformer = TfidfTransformer()
print(transformer)

tfidf = transformer.fit_transform(x)
print(tfidf.toarray())

#tf_idf_TfidfVectorizer

TfidfVectorizer = TfidfVectorizer(min_df=0, analyzer='word',ngram_range=(1,2),stop_words='english')

vz  = TfidfVectorizer.fit_transform(corpus)
print(vz.shape)

tfidf_dic = dict(zip(TfidfVectorizer.get_feature_names(), TfidfVectorizer.idf_))
tfidf = pd.DataFrame(columns=['tfidf']).from_dict(dict(tfidf),orient='index')
tfidf.columns = ['tfidf']
tfidf.tfidf.hist(bins=25, figsize=(15,7))
