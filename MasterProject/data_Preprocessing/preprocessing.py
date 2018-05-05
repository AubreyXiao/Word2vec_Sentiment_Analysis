from os import listdir
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from collections import Counter
import re

#clean the doc
def convert_text_to_tokens(text):
    #1:remove tags ,split, lowercases->list
    tokens = remov_tags_lowercase_to_list(text)
    #4:remove the punctuation
    tokens = remov_punctuation(tokens)
    #5:get the token from the tokenization
    tokens1 = split_tokens_to_token(tokens)
    #6:remov_nonalpha
    tokens = remov_nonalpha(tokens1)
    #7:filter out the stopwords
    tokens = filt_out_stopwords(tokens)
    #8:filter out the short words
    tokens = filt_out_shortwords(tokens)
    #9: do-stemming
    #tokens = do_stemming(tokens)

    return tokens

#1:remove tags from the text
def remov_tags_lowercase_to_list(text):
    #print(text)
    # 1:remove tags from the text
    text = re.sub("<.*?>", "\n", text)
    #print("1:remove the tags")
    #print(text)
    # 2:split the sentences into a list
    tokens = text.split("\n")
    #print("2:split the sentences into a list")
    #print(tokens)
    # 3:lowercase
    tokens = [token.lower() for token in tokens]
    #print("3:lowercase")
    #print(tokens)

    return tokens

# 4:remove the punctuations
def remov_punctuation(tokens):
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens if w != '']
    #print("4:remove punctuations")
    #print(tokens)

    return tokens


# 5:split the tokens
def split_tokens_to_token(tokens):
    # 5:tokenize and remove those are not alphabate
    tokens1 = []
    for token in tokens:
        #print(token)
        #print("\n")
        token2 = token.split()
        tokens1 += token2
        # token2 = word_tokenize(token)
        # tokens1 += token2
    #print("5:split the tokens to token list !!!!!!")
    #print(tokens1)
    #print(len(tokens1))

    return tokens1


#6:remove_nonalpa
def remov_nonalpha(tokens):
    tokens = [word1 for word1 in tokens if word1.isalpha()]
    #print("5:remove alphabate:")
    #print(tokens)

    return tokens

#7:filter out stopwords
def filt_out_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    # #print("6:remove stopwors:")
    # print(tokens)
    # print(len(tokens))

    return tokens

#8:filter out the shortwords
def filt_out_shortwords(tokens):
    tokens = [w for w in tokens if len(w) > 1]
    # print("7:remove shortwords:")
    # print(tokens)
    # print(len(tokens))

    return tokens



#9:stemming the words and return a list of stemmed-tokens
def do_stemming(tokens):
    stemmer = SnowballStemmer('english')
    tokens1 = [stemmer.stem(t) for t in tokens]
    # print("8:stemmed words:")
    # print(tokens1)
    # print(len(tokens1))
    return tokens1


#------------------------IO------------------------------------

#1:load all docs in a directory
def process_files(directory, vocab, is_trian):
    for filename in listdir(directory):
       if not filename.endswith(".txt"):
           continue
       if not is_trian:
           break
       path = directory + '/' + filename
       text = load_file(path)
       convert_text_to_tokens(text)
       add_doc_to_vocab(path, vocab)

#2:load the file and return the text
def load_file(filename):
    file = open(filename,'r')
    text = file.read()
    file.close()
    return text


#3:load doc and add to vocab
def add_doc_to_vocab(filename,vocab):
    text = load_file(filename)
    tokens = convert_text_to_tokens(text)
    vocab.update(tokens)


#save a list to a file
def save_to_file(token,filename):
    data = '\n'.join(token)
    file = open(filename,'w')
    file.write(data)
    file.close()


#should pass a vocab
def shrink_vocab(vocab, k):
    min_occurrence = k
    tokens = [ w for w, k in vocab.items() if k>=min_occurrence ]
    print(vocab.most_common(50))
    return tokens

#-------------------------main---------------------------------------------------

