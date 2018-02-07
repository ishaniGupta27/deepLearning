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


x_train_real = [] # input word--Center word in our case
y_train_real = [] # output word---> nb word ! the one forming a tuple
for data_word in data:
    x_train_real.append(to_one_hot(word2int[ data_word[0] ], vocabSize))
    y_train_real.append(to_one_hot(word2int[ data_word[1] ], vocabSize))


# convert them to numpy arrays
x_train = np.asarray(x_train_real[0:1])
y_train = np.asarray(y_train_real[0:1])

print 'Shape of training set input'
print x_train.shape #this is just one word represented by vocab size =7 
print x_train

print 'Shape of training set output'
print y_train.shape #this is just one word represented by vocab size =7 


def softmax_function(x): #X is a matrix 
    """
    Computing the softmax function for each row of the input x to be used doe skip gram model.

    Argument:
    x -- A numpy matrix of shape (n,m)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (n,m)
     """
   
    # Applying exp() element-wise to x.
    max_x=np.max(x, axis=1)[:,np.newaxis]
    #For numerical stability , finding softmax of (x-maxx)
    np.reshape(max_x,(max_x.shape[0],1))
    print max_x.shape
    x_exp = np.exp(x-max_x)#this will make all the elemnts change to there exponents

    # Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis = 1, keepdims = True).
    x_sum = np.sum(x_exp,axis=1,keepdims=True)
    
    # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting.
    s = x_exp/x_sum ##broadcating make things easier !
    
    return s


def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples) ..... 2*400
    Y -- labels of shape (output size, number of examples)...... 1*400
    
    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    
    n_x = X.shape[1] # size of input layer i.e vocabsize for now
    n_h = 5 #this is size of hidden layer
    n_y = Y.shape[1] # size of output layer i.e vocab size for now
    
    return (n_x, n_h, n_y)


def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer = 7
    n_h -- size of the hidden layer = 5
    n_y -- size of the output layer = context window * 7
    
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    W2 -- weight matrix of shape (n_y, n_h)
    """
    
    W1 = np.random.randn(n_x,n_h)*0.01 #this is vocab * dimen==> 7 * 5 
    W2 = np.random.randn(n_h,n_y)*0.01 #this is dimen * vocab ===> 5*7
   
   
    
    assert (W1.shape == (n_x, n_h)) 
    assert (W2.shape == (n_h, n_y))
   
    
    parameters = {"W1": W1,
                  "W2": W2,}    
    return parameters

def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    cache: #to complete
    """
    # Retrieving each parameter from the dictionary "parameters"
    
    W1 = parameters["W1"] 
    W2 = parameters["W2"]
   
    
    
    # Implementing Forward Propagation to calculate output vectors
    
    Z1 = np.dot(X,W1)  #X---vocab*no. of examples (1*7), W1:: vocab * dimen = 7*5
    A1 = Z1 ## this is dimen*no.of examples

    print "Shape of A1"
    print A1.shape  ##this is 1*5

    ##h---------------CONTEXT WINDOW-------------------------------------------------
    Z2_first = np.dot(A1,W2) #W2::: dimension * vocab==> 5*7, A1:: 1* dimension= 1*5
    A2_first = softmax_function(Z2_first)
    print "Shape of A2_first"
    print A2_first.shape  ##this is 1*5
   
    Z2_second = np.dot(A1,W2) #W2::: dimension * vocab==> 5*7, A1:: 1* dimension= 1*5
    A2_second = softmax_function(Z2_second)
    print "Shape of A2_second"
    print A2_first.shape  ##this is 1*5


    cache = {"Z1": Z1,
             "A1": A1,
             "Z2_first":Z2_first,
             "A2_first":A2_first,
             "Z2_second":Z2_first,
             "A2_second":A2_first}
    
    return cache
'''
def compute_cost(predicted,outputVectors): 

	probabilities = softmax(predicted.dot(outputVectors.T))
    cost = -np.log(probabilities[target])
'''
def stocgraddesc():


def skipgram_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    
    n_x = layer_sizes(X, Y)[0] #==vocabsize=7
    n_y = layer_sizes(X, Y)[2] #==vocabsize=7
    
    
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    cache = forward_propagation(X, parameters)
    #this is getting hardcoded
    

    ##this is part of backprop
    target1=to_one_hot(word2int["is"],vocabSize) #expected Output
    print "target1"
    print target1

    target2=to_one_hot(word2int["the"],vocabSize)
    print "target2"
    print target2

    error1=target1-cache["A2_first"]

    error2=target2-cache["A2_second"]

    errorvector=error1+error2;

    print errorvector

    
    


skipgram_model(x_train,y_train,5,num_iterations=10000, print_cost=True)










