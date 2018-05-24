import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure,show,output_notebook
from gensim.models import Word2Vec

#load the model
model = Word2Vec.load("300features_40minwords_10window")

word_vectors = model.wv

print(len(word_vectors.vocab))
print(word_vectors.index2word[0])
print(word_vectors.index2word[len(word_vectors.vocab)-1])
print(word_vectors.vocab['of'].index)

print(word_vectors.vocab.keys())

print(word_vectors.vocab)
print(word_vectors.vector_size)

print(word_vectors.vectors)



print(word_vectors.most_similar(positive=['woman', 'king'], negative=['man']))
print(word_vectors.similarity('woman','man'))


#define chart
output_notebook()
plot_tfidf = bp.figure(plot_width = 700,plot_height = 600, title = "map of 10K word vectors", tools = "pan, wheel_zoom, box_zoom, reset, hover, previewsave", x_axis_type = None, y_axis_type = None, min_border = 1)

#getting a list of word vectors, limit to 1000.each is of 300 dimensions

