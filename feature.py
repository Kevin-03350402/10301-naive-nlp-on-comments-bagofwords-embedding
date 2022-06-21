import csv
import numpy as np


import sys
import math


VECTOR_LEN = 300   # Length of word2vec vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and word2vec.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################





def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_dictionary(file):
    """
    Creates a python dict from the model 1 dictionary.

    Parameters:
        file (str): File path to the dictionary for model 1.

    Returns:
        A dictionary indexed by strings, returning an integer index.
    """
    dict_map = np.loadtxt(file, comments=None, encoding='utf-8',
                          dtype=f'U{MAX_WORD_LEN},l')
    return {word: index for word, index in dict_map}


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the word2vec
    embeddings.

    Parameters:
        file (str): File path to the word2vec embedding file for model 2.

    Returns:
        A dictionary indexed by words, returning the corresponding word2vec
        embedding np.ndarray.
    """
    word2vec_map = dict()
    with open(file) as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            word2vec_map[word] = np.array(embedding, dtype=float)
    return word2vec_map


train_input= load_tsv_dataset(sys.argv[1])

validation_input = load_tsv_dataset(sys.argv[2])

test_input = load_tsv_dataset(sys.argv[3])

dict_input = load_dictionary(sys.argv[4])

feature_dictionary_input = load_feature_dictionary(sys.argv[5])

feature_flag = int(sys.argv[9])


def Stringlist(string):
    return(list(string.split(" ")))

def bagofword(input, dict_input):
    length = len(input)
    #go through all the words
    wordcomments = []
    labels = []
    for i in range(0,length):
        #get the comment with 01
        rawcomment = input[i]
        labels.append(rawcomment[0])
        # get out of the 01
        wordcomments.append(rawcomment[1])
    # bagofword returns a vector with length same as the dictionary
    # make a new list, contains the splitted comments
    splitcomment = []

    # go through the entire words vect
    for j in range(0,len(wordcomments)):
        comment = wordcomments[j]
        line = Stringlist(comment)
        splitcomment.append(line)
    
    bagvectorlength = len(dict_input)
    res = []
    for k in range(0, len(splitcomment)):
        bagvector = [0]*(bagvectorlength)
        #reserve a place for y
        wordlist  = splitcomment[k]
        ylab = int(labels[k])
        
        # get the comment
        for wi in range(0,len(wordlist)):
            # the word in the comment 
            word =  wordlist[wi]
            if (word in dict_input):
                bagpos = dict_input[word]
                bagvector[bagpos] = 1
        
        bagvector = [ylab] + bagvector

        res.append(bagvector)
    
    return res
        


def  embedding(input, feature_dictionary_input):
    length = len(input)
    #go through all the words
    wordcomments = []
    labels = []
    for i in range(0,length):
        #get the comment with 01
        rawcomment = input[i]
        labels.append(rawcomment[0])
        # get out of the 01
        wordcomments.append(rawcomment[1])
    # bagofword returns a vector with length same as the dictionary
    # make a new list, contains the splitted comments
    splitcomment = []

    # go through the entire words vect
    for j in range(0,len(wordcomments)):
        comment = wordcomments[j]
        line = Stringlist(comment)
        splitcomment.append(line)
    
    res = []

    for k in range(0, len(splitcomment)):
        embeddingvector = np.zeros(301)
        
        #reserve a place for y
        wordlist  = splitcomment[k]
        ylab = int(labels[k])
        ylab = '%.6f' % ylab
        ylab = float(ylab)
        
        # get the comment
        counter = 0
        for wi in range(0,len(wordlist)):
            
            # the word in the comment 
            word =  wordlist[wi]
            if (word in feature_dictionary_input):
                counter+=1
                embedding = feature_dictionary_input[word]
                embedding = np.insert(embedding, 0, 0)
                embeddingvector+=embedding 
        embeddingvector = embeddingvector/counter
        # add the labels to beginning

        embeddingvector[0] = ylab
        cl = embeddingvector.tolist()
        cl = [round(num, 6) for num in cl]
        #get the idea of rounding every element in list from stackoverflow
        res.append(cl)

    return res

formatted_train_out = sys.argv[6]
formatted_validation_out = sys.argv[7]
formatted_test_out = sys.argv[8]

if (feature_flag == 1):
    trainm1 = bagofword(train_input, dict_input)
    validationm1 = bagofword(validation_input, dict_input)
    testm1 = bagofword(test_input, dict_input)

    trainm1 = np.array(trainm1)
    np.savetxt(formatted_train_out, trainm1, delimiter='\t', fmt='%d')

    validationm1 = np.array(validationm1)
    np.savetxt(formatted_validation_out,validationm1, delimiter='\t', fmt='%d')

    testm1 = np.array(testm1)
    np.savetxt(formatted_test_out,testm1, delimiter='\t', fmt='%d')

if (feature_flag == 2):
    
    trainm2 = embedding(train_input, feature_dictionary_input)

    validationm2 = embedding(validation_input, feature_dictionary_input)
    testm2 = embedding(test_input, feature_dictionary_input)

    trainm2 = np.array(trainm2)

    np.savetxt(formatted_train_out, trainm2, delimiter='\t', fmt='%.6f')

    validationm2 = np.array(validationm2)
    np.savetxt(formatted_validation_out,validationm2, delimiter='\t', fmt='%.6f')

    testm2 = np.array(testm2)
    np.savetxt(formatted_test_out,testm2, delimiter='\t', fmt='%.6f')
