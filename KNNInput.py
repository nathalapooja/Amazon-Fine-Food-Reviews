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
stop_Words = []# Stopwords
g_dataset = [] # For new dataset 

# varaible for regular expressions
reg_tokens = ""
in_Review = []


# initialize the system and removes stopwords from dataset
def init_System():
    print "Stop words preparation for removing "
    stop_word = set(stopwords.words('english') + punctuation + ['rt', 'via', 'i\'m', 'us', 'it'])
    for x in stop_word:
        stop_Words.append(stemmer.stem(lemmatiser.lemmatize(x, pos="v")))

# wordTokenize takes out each word and consider it as tokens
def wordTokenize(s):
    return reg_tokens.findall(s)

# Here we remove the emoticons, special characters if present and does stemming and lemmatizing of each word
def data_Preprocess(stg, lowercase=True):
    stg = re.sub('[^a-zA-Z0-9\n\.]', ' ', stg)
    gen_tokens = wordTokenize(stg)
    if lowercase:
        gen_tokens = [token if emoticon_re.search(token) else stemmer.stem(lemmatiser.lemmatize(token.lower(), pos="v")) for
                  token in gen_tokens]
    return gen_tokens


# string_Processing takes row given and converts it to terms list. Takes each term and removes the stopwords 
def string_Processing(string):
    terms_stop = [term for term in data_Preprocess(string) if term not in stop_Words and len(str(term)) > 1]
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
        str1 = ' '.join(temp)
        temp = [str1]
        temp.append(row["rating"])
        cnvrted_Reviews.append(temp)
    return cnvrted_Reviews

# Function is used to send each row for preprocessing
def convertReviewsUser(reviews):
    cnvrted_Reviews=[]
    for a in reviews:
        cnvrted_Reviews.append(string_Processing(str(a).lower()))
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

# To calculate the hamming distance between each two sentences
def haming_DistanceCal(str1, str2):
    diffs = 0

    str1 = str1[0]
    str2 = str2[0]

    if len(str1) != len(str2):
        return max(len(str1), len(str2))
    for ch1, ch2 in zip(str1, str2):
        if ch1 != ch2:
            diffs += 1
    return diffs

# To calculate edit distance between every two sentences
def modify_Distance(s1, s2):
    s1 = s1[0]
    s2 = s2[0]

    m = len(s1) + 1
    n = len(s2) + 1

    tbl = {}
    for i in range(m): tbl[i, 0] = i
    for j in range(n): tbl[0, j] = j
    for i in range(1, m):
        for j in range(1, n):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            tbl[i, j] = min(tbl[i, j - 1] + 1, tbl[i - 1, j] + 1, tbl[i - 1, j - 1] + cost)

    return tbl[i, j]

# Fucntion to get all the neighbors whose distance is less
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

# initializing NLTK functions for preprocessing
reg_tokens = re.compile(r'(' + '|'.join(stg_regex) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + stg_emoticons + '$', re.VERBOSE | re.IGNORECASE)
punctuation = list(string.punctuation)
stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()
init_System()

# Main function starts here
if __name__ == "__main__":
    sc = SparkContext(appName="KNNUserInput")

    load_csvFile("Reviews.csv")#call the load_csvFile function which appends column 9 and column 6 to the new dataset named g_dataset
    trainDataFtReviews_pd = pd.DataFrame(g_dataset, columns=["review", "rating"])# constructs a dataframe from review and rating colums 
    targetData_Review = trainDataFtReviews_pd['review']# only review column is taken for further processing 
    trainData_Reviews = convertByRow_Reviews(trainDataFtReviews_pd)# This step calls covertByRow_Reviews method which does all data preprocessing steps

    k = 25

    train_Data = sc.parallelize(trainData_Reviews)# An RDD named train_Data is created from the preprocessed data

    print "User inputs are: "
    in_Review  = open("input.txt").readlines()
    print in_Review 
    outpu = open('output.txt', 'w')
    i = 0
    outpu.write("id \t\t review  \t\t\t\t\t rating \n")
    # Calculate the K nearest neighbors, classify based on the maximum class and output the predicted rating
    for inpeach in in_Review :

        ipi = inpeach.strip()
        userReview = [(ipi,3)]
        print userReview
        test_Data = pd.DataFrame(userReview, columns=["review", "rating"])
        userReviews = convertReviewsUser(test_Data)
        print userReviews;
        test = sc.parallelize(userReviews)

        nearestNeigbours = test.cartesian(train_Data).map(lambda (test, train_Data): (test, [haming_DistanceCal(test, train_Data)] + train_Data)).sortBy(lambda (test, train_Data): train_Data[0]).map(lambda (key, value): (tuple(key), value)).groupByKey().map(lambda x: (x[0], list(x[1]))) # building the kdd trees

        nearestNeigbours2 = nearestNeigbours.map(lambda (test, train_Data): (test, train_Data[0:k]))#consider the first k neighbours 

        nearestNeigbours3 = nearestNeigbours2.map(lambda (test, train_Data): (test, getNeigh(train_Data)))#Assign to the class whose k value is


        testsetfinal = nearestNeigbours3.keys().collect()
        predicted = nearestNeigbours3.values().collect()
        i = i + 1
        outpu.write(repr(i) + "  " + repr(ipi) + "  " + repr(predicted[0]) + "\n")
