from gensim.models import Word2Vec



class model():


    #1:get model
    def get_model(self,modelname):

        dir = "/Users/xiaoyiwen/Desktop/Sentiment_Analysis_word2vec/MasterProject/data_Preprocessing/model/word2vec_model/"
        modelname = dir + modelname
        model = Word2Vec.load(modelname)
        return model




#----main-----
