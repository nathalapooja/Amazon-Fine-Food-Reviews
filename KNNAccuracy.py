import sys
import random
import math
import operator
import os

import csv

from pyspark import SparkContext

import numpy as np  
from numpy import *
import numpy.ma

from random import randint
import math

import re
import string
import pandas as pd
from collections import Counter

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.probability import FreqDist as nF
from nltk.corpus import stopwords
from nltk import bigrams
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

from operator import add

# Varaibles which are Lists in python
stpwrds = []# Stopwords
g_dataset = [] # For new dataset generated
dec_Attributes = []

# varaible for regular expressions
reg_tokens = ""

# initialize the system and removes stopwords from dataset

def init_System():
    print "Stop words preparation for removing "
    stop_word = set(stopwords.words('english') + punctuation + ['rt', 'via', 'i\'m', 'us', 'it'])
    for x in stop_word:
        stpwrds.append(stemer.stem(lematiser.lemmatize(x, pos="v")))

# wordTokenize takes out each word and consider it as tokens
def wordTokenize(s):
    return reg_tokens.findall(s)

# Here we remove the emoticons, special characters if present and does stemming and lemmatizing of each word
def data_Preprocess(stg, lowercase=True):
    stg = re.sub('[^a-zA-Z0-9\n\.]', ' ', stg)
    gen_tokens = wordTokenize(stg)
    if lowercase:
        gen_tokens = [token if emoticon_re.search(token) else stemer.stem(lematiser.lemmatize(token.lower(), pos="v")) for
                  token in gen_tokens]
    return gen_tokens

# string_Processing takes row given and converts it to terms list. Takes each term and removes the stopwords 
def string_Processing(string):
    terms_stop = [term for term in data_Preprocess(string) if term not in stpwrds and len(str(term)) > 1]
    return terms_stop

# load_csvFile reads the csv file,trims the header and takes only score and text columns and appends to the dataset names g_dataset.
def load_csvFile(pathOfFile):
    print "Loading the dataset..."
    read_File = open(pathOfFile, "r")
    reader_csv = csv.reader(read_File, dialect='excel')
    next(reader_csv, None)
    for row in reader_csv:
        temp = (row[9], row[6])
        #temp = (row[0], row[1])
        g_dataset.append(temp)

# convertByRow_Reviews function sends each row for preprocessing and then generates the new dataset from the preprocessed rows
def convertByRow_Reviews(reviews):
    cnvrted_Reviews = []
    for index, row in reviews.iterrows():
        temp = string_Processing(str(row["review"]).lower())
        string1 = ' '.join(temp)
        temp = [string1]
        temp.append(row["rating"])
        cnvrted_Reviews.append(temp)
    return cnvrted_Reviews

# definition for emoticons string which is useful in removing if any exists
stg_emoticons = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

# Regular expression for non alphanumeric characters
stg_regex = [
    r'<[^>]+>',  # HTML tags
    r"(?:[a-z][a-z\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

# Calculating tha hamming dist bn every two lines
def haming_DistanceCal(string1, string2):
    d = 0

    string1 = string1[0]
    string2 = string2[0]

    if len(string1) != len(string2):
        return max(len(string1), len(string2))
    for ch1, ch2 in zip(string1, string2):
        if ch1 != ch2:
            d += 1
    #print "Ending **** Entering hamm distance 2 with **** Ending " + string1 + " : "+ string2
    return d

# calculating edit distance bn each 2 lines
def modify_Distance(s1, s2):
    s1 = s1[0]
    s2 = s2[0]

    m = len(s1) + 1
    n = len(s2) + 1

    table = {}
    for i in range(m): table[i, 0] = i
    for j in range(n): table[0, j] = j
    for i in range(1, m):
        for j in range(1, n):
            cost_value = 0 if s1[i - 1] == s2[j - 1] else 1
            table[i, j] = min(table[i, j - 1] + 1, table[i - 1, j] + 1, table[i - 1, j - 1] + cost_value)

    return table[i, j]

# getting all those neighbors who have less dist
def getNeigh(neighbors):
    class_ForVotes = {}
    for x in range(len(neighbors)):
        rspons = neighbors[x][-1]
        if rspons in class_ForVotes:
            class_ForVotes[rspons] += 1
        else:
            class_ForVotes[rspons] = 1
    votes_Sorted = sorted(class_ForVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return votes_Sorted[0][0]

# calculating the accuracy value
def accuracy_KNN(testing_set, predicted_value):
    right_value = 0
    for x in range(len(testing_set)):
        if testing_set[x][-1] == predicted_value[x]:
            right_value += 1
    return (right_value / float(len(testing_set))) * 100.0

# for preprocesing, initialize NLTK functions 
reg_tokens = re.compile(r'(' + '|'.join(stg_regex) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + stg_emoticons + '$', re.VERBOSE | re.IGNORECASE)
punctuation = list(string.punctuation)
stemer = PorterStemmer()
lematiser = WordNetLemmatizer()
init_System()

# Main func 
if __name__ == "__main__":
    sc = SparkContext(appName="KNN")# creating a sparkcontext object with application name "KNN"

    load_csvFile("Reviews.csv")#call the load_csvFile function which appends column 9 and column 6 to the new dataset named g_dataset
    trainDataFtReviews_pd = pd.DataFrame(g_dataset, columns=["review", "rating"])# constructs a dataframe from review and rating colums 
    targetData_Review = trainDataFtReviews_pd['review']# only review column is taken for further processing 

    print "Preprocesseing the Data"
    trainData_Reviews = convertByRow_Reviews(trainDataFtReviews_pd)# This step calls covertByRow_Reviews method which does all data preprocessing steps

    yx_RDD = sc.parallelize(trainData_Reviews)# An RDD named yx_RDD is created from the preprocessed data
    train_Data, test_Data = yx_RDD.randomSplit([0.6, 0.4])# Splitting the RDD into training and test where training is used for building the model and test is used for finding the accracy of algorithm
    #test_Data = sc.parallelize(['This  is great stuff.  Made some really tasty banana bread.  Good quality and lowest price in town.'])

    k = 10 #k value is chosen as 10
    #print test_Data.collect()[0]

    print "Preparing the KNN model"

    nearestNeigbours = test_Data.cartesian(train_Data).map(lambda (test_Data, train_Data): (test_Data, [haming_DistanceCal(test_Data, train_Data)] + train_Data)).sortBy(lambda (test_Data, train_Data): train_Data[0]).map(lambda (key, value): (tuple(key), value)).groupByKey().map(lambda x: (x[0], list(x[1]))) # building the kdd tree



    print "Done preparing the model"


	# Calculating k nearst neighbours, classifying them on basis of the maximum class the neighbours belong to, and then calculating the accuracy
    while (k <= 50):

        nearestNeigbours2 = nearestNeigbours.map(lambda (test_Data, train_Data): (test_Data, train_Data[0:k]))#consider the first k neighbours 
        nearestNeigbours3 = nearestNeigbours2.map(lambda (test_Data, train_Data): (test_Data, getNeigh(train_Data)))#Assign to the class whose k value is

        final_test_Data_set = nearestNeigbours3.keys().collect()
        predicted = nearestNeigbours3.values().collect()

        accuracy = accuracy_KNN(final_test_Data_set, predicted)
        print "====================================================================================================="
        print('Accuracy calculated for KNN with k '+ repr(k)+ ' is ' + repr(accuracy) + '%')
        print "====================================================================================================="

        k=k+10
