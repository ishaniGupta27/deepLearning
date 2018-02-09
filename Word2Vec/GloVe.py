import numpy as np 
import json
from random import shuffle

with open('config.json') as json_data_file:
    ConfigJsondata = json.load(json_data_file)
#------------------------------------------------------------------------------------------##
#------------------------------GloVe start-------------------------------------------------##
#------------------------------------------------------------------------------------------##
 '''
 Pennington et al. argue that the online scanning approach used by word2vec (Shallow window method)
 is suboptimal since it doesn’t fully exploit statistical information regarding word co-occurrences.

In the Hyperspace Analogue to Language (HAL) and derivative approaches, square matrices are used where the rows and
 columns both correspond to words and the entries correspond to the number of times a given word occurs in the context 
 of another given word (a co-occurrence matrix). One problem that has to be overcome here is that frequent words contribute 
 a disproportionate amount to the similarity measure and so some kind of normalization is needed.
'''
#------------------------------------------------------------------------------------------##
#------------------------------GloVe end --------------------------------------------------##
#------------------------------------------------------------------------------------------##
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
print "The Vocab Size is :::"
print vocabSize
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



def run_iter(vocab, data, learning_rate=0.05, x_max=100, alpha=0.75):
    """
    Run a single iteration of GloVe training using the given
    cooccurrence data and the previously computed weight vectors /
    biases and accompanying gradient histories.
    `data` is a pre-fetched data / weights list where each element is of
    the form
        (v_main, v_context,
         b_main, b_context,
         gradsq_W_main, gradsq_W_context,
         gradsq_b_main, gradsq_b_context,
         cooccurrence)
    as produced by the `train_glove` function. Each element in this
    tuple is an `ndarray` view into the data structure which contains
    it.
    See the `train_glove` function for information on the shapes of `W`,
    `biases`, `gradient_squared`, `gradient_squared_biases` and how they
    should be initialized.
    The parameters `x_max`, `alpha` define our weighting function when
    computing the cost for two word pairs; see the GloVe paper for more
    details.
    Returns the cost associated with the given weight assignments and
    updates the weights by online AdaGrad in place.
    """

    global_cost = 0

    # We want to iterate over data randomly so as not to unintentionally
    # bias the word vector contents
    shuffle(data)

    for (v_main, v_context, b_main, b_context, gradsq_W_main, gradsq_W_context,
         gradsq_b_main, gradsq_b_context, cooccurrence) in data:

        weight = (cooccurrence / x_max) ** alpha if cooccurrence < x_max else 1

        # Compute inner component of cost function, which is used in
        # both overall cost calculation and in gradient calculation
        #
        #   $$ J' = w_i^Tw_j + b_i + b_j - log(X_{ij}) $$
        cost_inner = (v_main.dot(v_context)
                      + b_main[0] + b_context[0]
                      - log(cooccurrence))

        # Compute cost
        #
        #   $$ J = f(X_{ij}) (J')^2 $$
        cost = weight * (cost_inner ** 2)

        # Add weighted cost to the global cost tracker
        global_cost += 0.5 * cost

        # Compute gradients for word vector terms.
        #
        # NB: `main_word` is only a view into `W` (not a copy), so our
        # modifications here will affect the global weight matrix;
        # likewise for context_word, biases, etc.
        grad_main = weight * cost_inner * v_context
        grad_context = weight * cost_inner * v_main

        # Compute gradients for bias terms
        grad_bias_main = weight * cost_inner
        grad_bias_context = weight * cost_inner

        # Now perform adaptive updates
        v_main -= (learning_rate * grad_main / np.sqrt(gradsq_W_main))
        v_context -= (learning_rate * grad_context / np.sqrt(gradsq_W_context))

        b_main -= (learning_rate * grad_bias_main / np.sqrt(gradsq_b_main))
        b_context -= (learning_rate * grad_bias_context / np.sqrt(
                gradsq_b_context))

        # Update squared gradient sums
        gradsq_W_main += np.square(grad_main)
        gradsq_W_context += np.square(grad_context)
        gradsq_b_main += grad_bias_main ** 2
        gradsq_b_context += grad_bias_context ** 2

    return global_cos
#------------------------------------------------------------------------------------------##
#------------------------------Train Glove start-------------------------------------------##
#------------------------------------------------------------------------------------------##

def trainGloveModel(vocab, cooccurrences, dimen=100,iterations=25, **kwargs):
    """
    Training my Glove Model

    Argument:
    vocab -- int of the word to be represented by one hot vector
    cooccurrences --- Vocabulury Size
    vector_size --- Vocabulury Size
    iterations --- Vocabulury Size
    kwargs --- Vocabulury Size
    Returns:
    temp -- a 1*v vector
     """
    W = ((np.random.rand(vocabSize * 2, dimen) - 0.5)/ float(dimen + 1)) ##size---> (vocabSize*2) * dimen
    biasTerm = ((np.random.rand(vocabSize * 2) - 0.5)/ float(dimen + 1)) ##
    
    gradient_squared = np.ones((vocabSize * 2, vectorSize),dtype=np.float64)
    gradient_squared_biases = np.ones(vocabSize * 2,dtype=np.float64)
    data = [(W[i_main], W[i_context + vocabSize],biases[i_main : i_main + 1],
             biases[i_context + vocabSize : i_context + vocabSize + 1],
             gradient_squared[i_main], gradient_squared[i_context + vocabSize],
             gradient_squared_biases[i_main : i_main + 1],
             gradient_squared_biases[i_context + vocabSize
                                     : i_context + vocabSize + 1],
             cooccurrence)
            for i_main, i_context, cooccurrence in cooccurrences]
    
    for i in range(iterations):
        logger.info("\tBeginning iteration %i..", i)

        cost = run_iter(vocab, data, **kwargs)

        logger.info("\t\tDone (cost %f)", cost)

        if iter_callback is not None:
            iter_callback(W)

    return W

##------------------------------------------------------------------------------------------##
##------------------------------Train Glove end---------------------------------------------##
##------------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------------##



#------------------------------------------------------------------------------------------##

#------------------------------------------------------------------------------------------##
#------------------------------LOADING TEST DATA start-------------------------------------##
#------------------------------------------------------------------------------------------##

corpus_raw="hot king  king .  king  terror . sex king   queen"
#TO CHECK :print word2int['zoo'] #not in order...does it matter??
#converting to lower case
corpus_raw=corpus_raw.lower()
#handling multiple sentences
raw_sentences=corpus_raw.split('.')##Split sentences to test and train here ..maybe??
sentDict=[] #these are the words in sentences
for sen in raw_sentences:
    sentDict.append(sen.split())

#Lets generate the training set according to window
cooccur_matrix=np.empty((vocabSize-1, vocabSize-1));
#------------------------------------------------------------------------------------------##
#------------------------------Building Co-Occurance Matrix start--------------------------##
#------------------------------------------------------------------------------------------##
WINDOW_SIZE = int(ConfigJsondata["hyperparameters"]["window_size"])#sliding n-gram window
for sentence in sentDict: #pick a sentence at a time
    for word_index, word in enumerate(sentence):
        for nb_word in sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1] :  #all the neighbour word for "word"
            if nb_word != word:
                cooccur_matrix[word2int[word],word2int[nb_word]]=cooccur_matrix[word2int[word],word2int[nb_word]]+1
                cooccur_matrix[word2int[nb_word],word2int[word]]=cooccur_matrix[word2int[nb_word],word2int[word]]+1


#------------------------------------------------------------------------------------------##
#------------------------------Building Co-Occurance Matrix end----------------------------##
#------------------------------------------------------------------------------------------##
          

#------------------------------------------------------------------------------------------##
#------------------------------LOADING TEST DATA end-------------------------------------##
#------------------------------------------------------------------------------------------##

W = trainGloveModel(vocabWordsDict, cooccur_matrix,iter_callback=25,vector_size=arguments.vector_size,iterations=arguments.iterations,learning_rate=arguments.learning_rate)




