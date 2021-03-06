import numpy as np 
import json
import zipfile
import sys

with open('config.json') as json_data_file:
    ConfigJsondata = json.load(json_data_file)

#------------------------------------------------------------------------------------------##
#------------------------------BUILDING THE VOCAB DICT start-------------------------------------##
#------------------------------------------------------------------------------------------##

#builing vocab --> setting vocab size and making hot vectors !
with open("vocab.txt","r") as f:
    data = f.readlines()
vocab_file  = open("vocab.txt", "r") #just wanna read the file
word2int={}#hash map
int2word={}#hash map

vocabWordsDict=[] #Array 

for word in data:
       if word!='.': #'.' is not to be considered as a word
           vocabWordsDict.append(word.split('\n')[0])


vocabWordsDict=set(vocabWordsDict) #to remove multiple occurances


vocabSize=len(vocabWordsDict)#How many words we have in our vocablury

for i,word in enumerate(vocabWordsDict):
   
   word2int[word] = i
   int2word[i] = word 
#print "The Vocab Size is :::"
#print vocabSize
#print word2int
#------------------------------------------------------------------------------------------##
#------------------------------BUILDING THE VOCAB DICT end---------------------------------##
#------------------------------------------------------------------------------------------##




#Function to change to_one_hot_vector------------------------------------------------------##
def to_one_hot(data_point_index, vocab_size):
    """
    Computing the on_hot_vector for the word.

    Argument:
    data_point_index -- int of the word to be represented by one hot vector
    vocab_size --- Vocabulury Size
    Returns:
    temp -- a 1*v vector
     """
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp

def hot_to_number(data_point_index):
    res=0
    it = np.nditer(data_point_index, flags=['f_index'])
    while not it.finished:
        if it[0] == 1.0:
            res=it.index
            break
        it.iternext()
        
    return res
#------------------------------------------------------------------------------------------##

#------------------------------------------------------------------------------------------##
#------------------------------LOADING TEST DATA start-------------------------------------##
#------------------------------------------------------------------------------------------##

corpus_raw="hot king  king .  king  terror . sex king   queen"
corpus_size=0
def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = f.read(f.namelist()[0]).split()
    return data

corpus_raw = read_data('text8.zip')
corpus_size=len(corpus_raw)
sys.stdout.write("\n Our Corpus size is-------------\n")
sys.stdout.write(str(len(corpus_raw))) #len(data)
sys.stdout.write("\n--------------------------------\n")
#print (corpus_raw[1:10]) #this data is clean data..no fulls ops nohhing

#corpus_raw = fileData.readlines()
#corpus_raw="hot king  king .  king  terror . sex king   queen"
#TO CHECK :print word2int['zoo'] #not in order...does it matter??
#converting to lower case
#corpus_raw=corpus_raw.lower()
#handling multiple sentences
#raw_sentences=corpus_raw.split('.')##Split sentences to test and train here ..maybe??
#sentDict=[] #these are the words in sentences
#for sen in raw_sentences:
    #sentDict.append(sen.split())
#print sentDict[1]
#Lets generate the training set according to window
#data=[] #it will have one center word and its neighbours for many center words
data = []
WINDOW_SIZE = int(ConfigJsondata["hyperparameters"]["window_size"])
#for sentence in sentDict: #pick a sentence at a time
for word_index, word in enumerate(corpus_raw):
    if(word_index>=WINDOW_SIZE & word_index<(corpus_size-10)):
        counter=0;
        while counter<WINDOW_SIZE:
            #0,1
            '''
            Sanity check
            print "word is..."
            print word 
            print word_index
            print "prev word is..."
            print corpus_raw[word_index-(counter)]
            print word_index-(counter)
            print "next word is..."
            print corpus_raw[word_index+counter+2]
            print word_index+counter+2
            '''
            if word in vocabWordsDict:
                if corpus_raw[word_index-(counter)] in vocabWordsDict:
                    #print "------------------"
                    #print corpus_raw[word_index-(counter)]
                    data.append([word, corpus_raw[word_index-(counter)]])# 2-0-1=1,2-1-1=0
                #else:
                    #data.append([word, "markedUn"])# marking unknown 
                if((word_index+counter+2)%10000==0):
                    sys.stdout.write("\n -------------------------------- \n")
                    sys.stdout.write(str(word_index+counter+2)) #len(data)
                    sys.stdout.write("\n --------------------------------\n")
                if (word_index+counter+2)<(corpus_size-2):
                    if corpus_raw[word_index+counter+2] in vocabWordsDict:
                        #print "------------------"
                        #print corpus_raw[word_index+counter+2]
                        data.append([word, corpus_raw[word_index+counter+2]])# 2+0+1=3,2+1+1=4
                #else:
                   # data.append([word, "markedUn"])# marking unknown 
                    
            counter=counter+1
            #for nb_word in sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1] :  #all the neighbour word for "word"
               # if nb_word != word:
                   # data.append([word, nb_word])
#data is having the tuple
#------------------------------------------------------------------------------------------##
#------------------------------LOADING TEST DATA end-------------------------------------##
#------------------------------------------------------------------------------------------##
# Now we have data in pairs.


#------------------------------------------------------------------------------------------##
#------------------------------DATA TO ONE HOT VECTOR start--------------------------------##
#------------------------------------------------------------------------------------------##

x_train_real = [] # input word--Center word in our case
y_train_real = [] # output word---> nb word ! the one forming a tuple
#print "-----------Length of the tuples----------------------"
sys.stdout.write("\n -------------------------------- \n")
sys.stdout.write(str(len(data))) #len(data)
sys.stdout.write("\n --------------------------------\n")
#print "-----------------------------------------------------"
for data_word in data:
    #print data_word
    x_train_real.append(to_one_hot(word2int[ data_word[0] ], vocabSize))
    y_train_real.append(to_one_hot(word2int[ data_word[1] ], vocabSize))


# convert them to numpy arrays
x_train = np.asarray(x_train_real)
y_train = np.asarray(y_train_real)

#print x_train
#print y_train
#------------------------------------------------------------------------------------------##
#------------------------------DATA TO ONE HOT VECTOR end----------------------------------##
#------------------------------------------------------------------------------------------##

#print 'Shape of training set input'
#print x_train.shape #this is just one word represented by vocab size =V

#print 'Shape of training set output'
#print y_train.shape #this is just one word represented by vocab size =V

#------------------------------------------------------------------------------------------##
#-----------------------------FUNCTIONS START HERE-----------------------------------------##
#------------------------------------------------------------------------------------------##

def softmax_function(x): #X is a matrix 
    """
    Computing the softmax function for each row of the input x to be used for skip gram model.

    Argument:
    x -- A numpy matrix of shape (n,m)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (n,m)
     """
   
    # Applying exp() element-wise to x.
    max_x=np.max(x, axis=1)[:,np.newaxis]
    #For numerical stability , finding softmax of (x-maxx)
    np.reshape(max_x,(max_x.shape[0],1))
    #print max_x.shape
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
    #print X.shape[]
    n_x = X.shape[1] # size of input layer i.e vocabsize for now
    n_h = (ConfigJsondata["hyperparameters"]["dimen"]) #this is size of hidden layer
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
    W1 = np.random.uniform(low=-0.5/n_h, high=0.5/n_h, size=(n_x, n_h))
    #W1 = np.random.randn(n_x,n_h)*0.08 #this is vocab * dimen==> V*H
    W2 = np.zeros(shape=(n_h, n_y)) #this is dimen * vocab ===> H*V
    #Is there any relation between W1 and W2
   
    
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
   
    #print "Shape of X1"
    #print X.shape  ##this is 1*5

    
    # Implementing Forward Propagation to calculate output vectors
    
    Z1 = np.dot(X,W1)  #X---no. of examples*vocab (1*7), W1:: vocab * dimen = 7*5
    A1 = Z1 ## this is dimen*1
    #print "Shape of A1"
    #print A1.shape  ##this is 1*5

    ##h---------------CONTEXT WINDOW-------------------------------------------------
    Z2_first = np.dot(A1,W2) #W2::: dimension * vocab==> 5*7, A1:: 1* dimension= 1*5
    ##----------------------------------------------------------------------------------
    ##---------------------NEGATIVE SAMPLING--------------------------------------------
    ##---------------------------------------------------------------------------------- 
    ##----------------------------------------------------------------------------------
    #The idea behind negative sampling is to reduce the dimansions of 
    #print "Shape of Z2_first"
    #print Z2_first.shape  ##this is 1*5
    A2 = softmax_function(Z2_first)
    #print "Shape of A2_first"
    #print A2_first.shape  ##this is 1*7
    '''
    c=0
    val=np.multiply(WINDOW_SIZE,2)
    #the idea is to reduce the
    A2 = np.empty((val, vocabSize))
    while c<val:
        A2[c]=A2_first; #this is the same value repeated all the time ! dimension:: c*vocab!
        c=c+1
    #print "Shape of A2 after loop"
    #print A2.shape  ##this is 4*7--> this is windowsize*vocabsize
    '''


    cache = {"X1":X,
             "Z1": Z1,
             "A1": A1,
             "A2": A2}
    
    return cache


def stg_update_parameters(parameters,errorArr,cache,learning_rate,dimen):
    '''
      errorArr: c*vocab
      W2: d*vocab-->(for w1 take transpose)
      x--1*vocab



    '''
    i=0
    j=0
    W1=parameters["W1"]
    W2=parameters["W2"]
    A1=cache["A1"]
    X1=cache["X1"]
    ##similar to be done for W1
    '''
    while i<dimen :
        while j< vocabSize:
            cLay=0;
            sum=0;
            while cLay<(WINDOW_SIZE*2):
                sum=sum+errorArr[cLay,j]*A1[:,i]
                cLay=cLay+1
            W2[i][j]=W2[i][j]-(learning_rate*sum)
            j=j+1
        i=i+1
    '''
    #vectorizing
    '''
    c=0
    val=np.multiply(WINDOW_SIZE,2)
    A1_moif = np.empty((val, dimen))
    while c<val:
        A1_moif[c]=A1;
        c=c+1
    '''
    W2=W2-np.transpose((learning_rate)*(np.dot(np.transpose(errorArr),A1)))
   
    
    ##similar to be done for W2
    '''
    c=0
    X1_moif = np.empty((val, vocabSize))
    while c<val:
        X1_moif[c]=X1;
        c=c+1
    '''
    #This is for which this whole thing is done for !
    '''
    print '------------------------------------------'
    print '----------------here 1--------------------------'
    print hot_to_number(X1)
    print '------------------------------------------'
    print '----------------here 1--------------------------'
    print W1[hot_to_number(X1-1),:] 
    print W1[hot_to_number(X1),:] #Sanity Check ! there should be change in central word!
    print '------------------------------------------'
    print '------------------------------------------'
    '''
    W1=W1-np.transpose((learning_rate)*(np.dot(np.transpose(np.dot(errorArr,np.transpose(W2))),X1)))
    '''
    print '------------------------------------------'
    print '---------------here 2---------------------------'
    print W1[hot_to_number(X1-1),:] #Sanity check
    print W1[hot_to_number(X1),:] #Sanity check
    print '------------------------------------------'
    print '------------------------------------------'
    '''

    parameters = {"W1": W1,
                  "W2": W2}
    return parameters

#------------------------------------------------------------------------------------------##
#-----------------------------FUNCTIONS END HERE-----------------------------------------##
#------------------------------------------------------------------------------------------##   
'''

def skipgram_model_dummy(X, Y, n_h):
    
    n_x = layer_sizes(X, Y)[0] #==vocabsize=7
    n_y = layer_sizes(X, Y)[2] #==vocabsize=7
    
    
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    cache = forward_propagation(X, parameters)
    #this is getting hardcoded
    

    ##this is part of backprop
    target = np.empty(((WINDOW_SIZE*2), vocabSize))
     #hardcoded ! ##need to be changed 
    target[0]=to_one_hot(word2int["is"],vocabSize) #expected Output
    target[1]=to_one_hot(word2int["the"],vocabSize)
    target[2]=to_one_hot(word2int["is"],vocabSize) #expected Output
    target[3]=to_one_hot(word2int["the"],vocabSize)
    print "target shape is "
    print target.shape


    errorArr=target-cache["A2"]
    print "error shape is"
    print errorArr.shape #--->this is 4*7

    print stg_update_parameters(parameters,errorArr,cache,0.1,n_h)

    Questions to ask:::

    1. How to decide context window? How to decide hidden layer dimensionality?#trial and test
    2. in stocgraddesc: what is hi? is the output of hideen layer?? How to use SQD update(slide 25)--Done !
    3. How the input has to be--> i mena the data?? Should it pick the cebter word and +- window and keep them together.--> Ask Richika
    4. Goal is to get hidden layer parameters for all the center words?--W1 ##Done
    3. how to train?? Split the data an and then cehck teh accuracy !

    '''
#------------------------------------------------------------------------------------------##
#-----------------------------MAIN FUNCTION START HERE-------------------------------------##
#------------------------------------------------------------------------------------------##   
def skipgram_model_loop(X, Y, n_h):
    
    n_x = layer_sizes(X, Y)[0] #==vocabsize=7
    n_y = layer_sizes(X, Y)[2] #==vocabsize=7
    
    
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    '''
    target = np.empty(((WINDOW_SIZE*2), vocabSize))
     #hardcoded ! ##need to be changed 
    target[0]=to_one_hot(word2int["king"],vocabSize) #expected Output
    target[1]=to_one_hot(word2int["hot"],vocabSize)
    target[2]=to_one_hot(word2int["king"],vocabSize) #expected Output
    target[3]=to_one_hot(word2int["hot"],vocabSize)
    print "target shape is "
    print target.shape
    print '------------------------------------------'
    print '------------------------------------------'
    print parameters["W1"]
    print '------------------------------------------'
    print '------------------------------------------'
    '''
    for word_indx in range(len(X)):
        ### START CODE HERE ###
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        # X --> its is no. of center words * vocabsize
        cache = forward_propagation(X[[word_indx],:], parameters)
        target = Y[[word_indx],:]#np.empty((1, vocabSize))
        errorArr=target-cache["A2"]
        #print "error shape is"
        #print errorArr.shape #--->this is 4*7
        parameters = stg_update_parameters(parameters,errorArr,cache,(ConfigJsondata["hyperparameters"]["learningRate"]),n_h)
        ##break;
        ### END CODE HERE ###
        
        

    return parameters

#------------------------------------------------------------------------------------------##
#-----------------------------MAIN FUNCTION START HERE-------------------------------------##
#------------------------------------------------------------------------------------------##   
    '''
    Questions to ask:::

    1. How to decide context window? How to decide hidden layer dimensionality?learnign rate?
    2. How to use SQD update(slide 25)
    3. How the input has to be--> i menan the data?? Should it pick the cebter word and +- window and keep them together.
    4. Goal is to get hidden layer parameters for all the center words?
    3. how to train??

    '''

    

#def mainFunc():
res=skipgram_model_loop(x_train,y_train,(ConfigJsondata["hyperparameters"]["dimen"]))["W1"]
np.savetxt('/output/vectors.txt', res)

