import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
import nltk
from nltk.corpus import wordnet as wn

my_path = os.path.abspath(__file__)[0:67]

def list_of_survey_words (df):
    '''
    Creates list of words used in questionaires without the commonly used prep-
    positions like "I", "the" etc.
    '''
    columns = df.columns.values[47:72]
    used_words = " ".join([i[5:-1] for i in columns])
    tokens = set(nltk.word_tokenize(used_words))
    to_remove = {'unknown', 'a', 'The', 'thought', 'out', 'due', 'were', 'be',
               'in', 'an', 'either', 'could', 'me', 'of', 'have', 'if', 'tell',
               ',', 'to', 'get', 'more', 'as', 'it', 'There', 'the', 'at',
               'upon', 'When', 'or', 'throughout', 'when', 'I', 'My', 'After',
               'seemed', 'that', 'my', 'would', 'was'}
    tokens -= to_remove
    
    return tokens, to_remove

def similarity (df, res_tokens, sur_tokens):
    """
    input: original scores on dataframe
    compare= word list to compare with
    against= 1 word to 
    output 
        most similar word
        word it's most similar with
        
    """
    to_return = []
    percentage = []
    word = []
    
    for i in res_tokens:
        for j in sur_tokens:
#            pdb.set_trace()
            word_1 = wn.synsets(i)
            word_2 = wn.synsets(j)
            if (not word_1 or not word_2):
                x=0
            elif (len(word_1)>=1 or len(word_2)>=1):
                x = word_1[0].path_similarity(word_2[0])
            if (x == None):
                x=0
            percentage.append(x)
            word.append(j)
            temp = list(zip(percentage,word))
        to_return.append(sorted(temp)[-1])
    return sorted(to_return)[-1]

def nlp_stats(df, word="", exclude=""):
    """
    The objective is to graph the statistics from the NLP package with the 
    scores of the State_con survey
    
    input: x (DataFrame object)
    word (String) of what to save the name of the plot of response length vs 
        scores
    outout: 
    """
# =============================================================================
#  State_con: 47->72
#  State_con total: 73
#     scale being measured (state conviction)

# =============================================================================
    x = df.as_matrix()
    responses = x[:,-1]
    res_length = [len(nltk.word_tokenize(i)) for i in responses]
    scores = x[:,73]
    if (len(exclude)!=0):
        mask = x[:,3] == exclude
        scores = scores[mask]
        responses = responses[mask]
        
    if (word == "unique"):
        #computes percentage of unique words by using set of list (unique)
        res_length = [len(set(nltk.word_tokenize(i)))/len(nltk.word_tokenize(i))
            for i in responses]
#        res_length = np.log(res_length).tolist()
    scores = scores.tolist()
#    scores = np.log(scores).tolist()
    xy_tuple = list(zip(res_length,scores))
    xy_tuple = sorted(xy_tuple, key=lambda x: x[0])
    plt.plot(*zip(*xy_tuple))
    plt.xlabel('response length')
    if (word == "unique"):
        plt.xlabel('Percentage of unique words')
    plt.ylabel('scores')
    plt.title('line plot')
    plt.savefig(my_path + "/figures/failed_nlptest"+word+"and"+exclude+".png")
    
# =============================================================================
#     #SIMILARITY of words in questionaire to responses
#     sur_tokens, to_remove = list_of_survey_words(df)
#     response_tokens = [set(nltk.word_tokenize(i)) for i in responses]
#     similarities = []
#     
# =============================================================================
# =============================================================================
#     for i in response_tokens:
#         similarities.append(similarity(df, i, sur_tokens))
#     pdb.set_trace()
# =============================================================================
    return 0