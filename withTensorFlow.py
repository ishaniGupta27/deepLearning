import numpy as np 
import tensorflow as tf 
corpus_raw="He is the king . The king is royal . She is the royal  queen"

#converting to lower case

corpus_raw=corpus_raw.lower()

#workng on creating an input and output tuple

#dictinary which translates words to integers

wordsDict=[] #Array 

for word in corpus_raw.split():
	if word!='.': #'.' is not to be considered as a word
	    wordsDict.append(word)


wordsDict=set(wordsDict) #to remove multiple occurances

word2int={}#hash map
int2word={}#hash map

vocabSize=len(wordsDict)#How many words we have in our vocablury

for i,word in enumerate(wordsDict):
   word2int[word] = i
   int2word[i] = word 


print word2int["queen"]

#Now we have multiple sentences

raw_sentences=corpus_raw.split('.')

sentDict=[]

for sen in raw_sentences:
	sentDict.append(sen.split())

print sentDict


#Lets generate the training set according to window

data=[] #it will have one center word and its neighbours for many center words

data = []
WINDOW_SIZE = 2
for sentence in sentDict: #pick a sentence at a time
    for word_index, word in enumerate(sentence):
        for nb_word in sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1] :  #all the neighbour word for "word"
            if nb_word != word:
                data.append([word, nb_word])


print "------Printing th tuple------"
print data

#Making every word represent as one-hot vector



def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp


x_train = [] # input word--Center word in our case
y_train = [] # output word---> nb word ! the one forming a tuple
for data_word in data:
    x_train.append(to_one_hot(word2int[ data_word[0] ], vocabSize))
    y_train.append(to_one_hot(word2int[ data_word[1] ], vocabSize))


# convert them to numpy arrays
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

print 'Shape of training set input'
print x_train.shape


print 'Shape of training set output'
print y_train.shape





