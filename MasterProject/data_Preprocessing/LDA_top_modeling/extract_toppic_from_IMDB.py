from MasterProject.data_Preprocessing.Datasets import get_data
import gensim
from gensim import corpora
from gensim.models import ldamodel
from gensim.models.ldamodel import LdaModel
#get train_data
data = get_data.Datasets()


def get_document_tokens(review):
    return data.LDA_preprocessing(review, do_stem=True)


def get_all_words_tokens(datsets):
    # clean the data
    all_words = []
    for review in datsets:
        tokens = get_document_tokens(review)
        all_words.append(tokens)
    return all_words


train_data  = data.get_train_data()


all_words = get_all_words_tokens(train_data["review"])
print(len(all_words))
#create a dictionary
dictionary = corpora.Dictionary(all_words)

doc_term_matrix = [dictionary.doc2bow(review) for review in all_words]

print(dictionary)
print(len(dictionary))
print(doc_term_matrix)


#create the object for LDA model
lda = gensim.models.ldamodel.LdaModel

ldamodel = lda(doc_term_matrix,num_topics=46,id2word=dictionary,passes=50)
print(ldamodel.print_topics(num_topics=46, num_words=5))