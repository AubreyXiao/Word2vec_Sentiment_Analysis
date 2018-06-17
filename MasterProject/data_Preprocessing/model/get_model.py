from gensim.models import Word2Vec


class model():


    #1:get model
    def get_model(self,modelname):

        model = Word2Vec.load(modelname)
        return model



