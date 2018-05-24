from gensim.models import Word2Vec
from gensim.models import KeyedVectors



#load the model
def word2vec_model_accuracy(model,questions):
   # model = Word2Vec.load(modelname)
    accuracy  = model.accuracy(questions)

    total_correct  = len(accuracy[-1]['correct'])
    total_incorrect = len(accuracy[-1]['incorrect'])

    total = total_correct + total_incorrect

    percent  = lambda a:a/total*100


    print('Total sentences: {}, Correct: {:.2f}%, Incorrect: {:.2f}%'.format(total,percent(total_correct),percent(total_incorrect)))

#load the file for evaluation
def read_evaluation_file(questions):
    evaluation = open(questions,'r').readlines()
    num_sections = len([l for l in evaluation if l.startswith(':')])
    #":" represent the separation
    num_sentences = len(evaluation)-num_sections

    print('Total evaluation sentences: {}'.format(num_sentences))


#load the goole as an example

#evaluate the given model
def evaluate_model(model,questions_words,questions_phrases):

    #load the questions_phrases.txt
    print("Evaluate the words")
    word2vec_model_accuracy(model,questions_words)

  #  print("Evaluate the phrases")
  #  word2vec_model_accuracy(model,questions_phrases)



#----------------evaluation----------------------
questions_words = 'questions-words.txt'
questions_phrases = 'questions-phrases.txt'
print("words!")
read_evaluation_file(questions_words)
print("phrases!")
read_evaluation_file(questions_phrases)



#test the google model
#word2vec_model_accuracy(google,questions)
google = KeyedVectors.load_word2vec_format('/Users/xiaoyiwen/Desktop/GoogleNews-vectors-negative300.bin', binary = True)
evaluate_model(google,questions_words,questions_phrases)



#evaluate 300features_40minwords_10window
model_300_40_10 = Word2Vec.load("/Users/xiaoyiwen/Desktop/Sentiment_Analysis_word2vec/MasterProject/data_Preprocessing/300features_40minwords_10window")
model1_300_40_10 = Word2Vec.load("/Users/xiaoyiwen/Desktop/Sentiment_Analysis_word2vec/word3vect_sentiment_analysis/300features_40minwords_10window")

model_300_40_20 = Word2Vec.load("/Users/xiaoyiwen/Desktop/Sentiment_Analysis_word2vec/MasterProject/data_Preprocessing/300features_40minwords_20window")
model1_300_40_20 = Word2Vec.load("/Users/xiaoyiwen/Desktop/Sentiment_Analysis_word2vec/word3vect_sentiment_analysis/300features_40minwords_20window")



#model_400_30_10 = Word2Vec.load("/Users/xiaoyiwen/Desktop/Sentiment_Analysis_word2vec/MasterProject/data_Preprocessing/400features_30minwords_10window")



model_400_40_10 = Word2Vec.load("/Users/xiaoyiwen/Desktop/Sentiment_Analysis_word2vec/MasterProject/data_Preprocessing/400features_40minwords_10window")
model1_400_40_10 = Word2Vec.load("/Users/xiaoyiwen/Desktop/Sentiment_Analysis_word2vec/word3vect_sentiment_analysis/400features_40minwords_10window")

#evaluate
evaluate_model(model_300_40_10,questions_words,questions_phrases)
evaluate_model(model1_300_40_10,questions_words,questions_phrases)


evaluate_model(model_300_40_20,questions_words,questions_phrases)
evaluate_model(model1_300_40_20,questions_words,questions_phrases)


evaluate_model(model_400_40_10,questions_words,questions_phrases)
evaluate_model(model1_400_40_10,questions_words,questions_phrases)






















