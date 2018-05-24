from gensim.models import Word2Vec

modelname = "300features_40minwords_10window"

model = Word2Vec.load(modelname)

print(model.wv.most_similar(positive=['woman','king'],negative=['man']))
print(model.wv.most_similar_cosmul(positive=['woman','king'],negative=['man']))


print(model.wv.most_similar(positive=['bigger','small'],negative=['big']))
print(model.wv.most_similar_cosmul(positive=['bigger','small'],negative=['big']))

print(model.wv.most_similar("awful"))
print(model.wv.most_similar("disgusting"))
print(model.doesnt_match("man woman child kitchen".split()))

print(model.syn0_lockf.shape)
print("\n")

modelname = "300features_40minwords_20window"

model = Word2Vec.load(modelname)

print(model.wv.most_similar(positive=['woman','king'],negative=['man']))
print(model.wv.most_similar_cosmul(positive=['woman','king'],negative=['man']))

print(model.wv.most_similar(positive=['bigger','small'],negative=['big']))
print(model.wv.most_similar_cosmul(positive=['bigger','small'],negative=['big']))
print(model.wv.most_similar("awful"))
print(model.wv.most_similar("disgusting"))

print("\n")

modelname = "400features_40minwords_10window"

model = Word2Vec.load(modelname)

print(model.wv.most_similar(positive=['woman','king'],negative=['man']))
print(model.wv.most_similar_cosmul(positive=['woman','king'],negative=['man']))

print(model.wv.most_similar(positive=['bigger','small'],negative=['big']))
print(model.wv.most_similar_cosmul(positive=['bigger','small'],negative=['big']))

print(model.wv.most_similar("awful"))
print(model.wv.most_similar("disgusting"))

print("\n")

modelname = "400features_40minwords_20window"

model = Word2Vec.load(modelname)

print(model.wv.most_similar(positive=['woman','king'],negative=['man']))
print(model.wv.most_similar_cosmul(positive=['woman','king'],negative=['man']))

print(model.wv.most_similar("awful"))
print(model.wv.most_similar("disgusting"))
