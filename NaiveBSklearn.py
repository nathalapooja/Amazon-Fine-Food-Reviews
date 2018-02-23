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

from sklearn.naive_bayes import MultinomialNB

# Varaibles which are Lists in python
stop_Words = []# Stopwords
g_dataset = [] # For new dataset 
dec_Attributes = []
# varaible for regular expressions
reg_tokens = ""


# initialize the system and removes stopwords from dataset
def init_System():
    print "Stop words preparation for removing "
    stop_word = set(stopwords.words('english') + punctuation + ['rt', 'via', 'i\'m', 'us', 'it'])#nltk
    for x in stop_word:
        stop_Words.append(stemmer.stem(lemmatiser.lemmatize(x, pos="v")))#nltk

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
    reader_csv = csv.reader(read_File, dialect='excel')#csv module
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
def cnvrtReviews_Library(reviews):
    cnvrted_Reviews = []
    for a in reviews:
        cnvrted_Reviews.append(string_Processing(str(a).lower()))
    return cnvrted_Reviews


# Function to prepare sparse matrix
def prep_SparseMatrix(cnvrted_Reviews):
    sparse_Matrix = []
    for cr in cnvrted_Reviews:
        newCr = [0] * len(dec_Attributes)# [ 0 0 0 0 0 ]
        for word in cr:
            if word in dec_Attributes:
                index = dec_Attributes.index(word)
                newCr[index] += 1
            else:
                pass
        sparse_Matrix.append(newCr)
    return sparse_Matrix # returns a matrix with values 1 whose index is index of dec_Attributes word which matches with the word in coverted reviews 

# To fit the model built by the scikit learn library
def model_Fitter(function_value):
    nb_functionvalue = str(function_value)
    nb_functionvalue = nb_functionvalue.partition("(")[0]
    function_value.fit(dataFeatures, targetRating) #  amodel is fitted for the sparsemtarix generated for tainREveiews and the train ratings
    s = f = 0
    predictedRating = list(function_value.predict(testDataFeatures))
    for i in range(len(predictedRating)):
        if predictedRating[i] == testRating[i]: # if the predicted rating matches the original rating increment the value of s by 1
            s += 1
        else:
            f += 1
    print "Rate prediction by " + nb_functionvalue + " is: " + str(float(s) / len(predictedRating) * 100.0)


# Function to get decision attrubites
def getDec_Attributes(cnvrted_Reviews):
    toCount = []
    for a in cnvrted_Reviews:
        toCount.append(" ".join(a))
    str1 = ""
    for a in toCount:
        str1 += "".join(a)
    x = Counter(str1.split(" "))
    for (k, v) in x.most_common(min(500, len(x))):
        dec_Attributes.append(k)

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



# initializing NLTK functions for preprocessing
reg_tokens = re.compile(r'(' + '|'.join(stg_regex) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + stg_emoticons + '$', re.VERBOSE | re.IGNORECASE)
punctuation = list(string.punctuation)
stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()
init_System()

# Main function starts here
if __name__ == "__main__":
    sc = SparkContext(appName="Naive-Bayes-sklearn")# creating a sparkcontext object with application name "Naive-Bayes-sklearn"

    load_csvFile("Reviews.csv")# call the load_csvFile function which appends column 9 and column 6 to the new dataset named g_dataset
    trainDataFtReviews_pd = pd.DataFrame(g_dataset, columns=["review", "rating"])# constructs a dataframe from review and rating colums #pandas
    targetData_Review = trainDataFtReviews_pd['review']# only review column is taken for further processing 

    print "Preprocessing the Data"
    trainData_Reviews = convertByRow_Reviews(trainDataFtReviews_pd)# This step calls covertByRow_Reviews method which does all data preprocessing steps

    yx_RDD = sc.parallelize(trainData_Reviews)# An RDD named yx_RDD is created from the preprocessed data#sparkcontext
    train_Data, test_Data = yx_RDD.randomSplit([0.6, 0.4])# Splitting the RDD into training and test where training is used for building the model and test is used for finding the accracy of algorithm

    targetRating = train_Data.map(lambda (x, y): y).collect()
    targetData_Review = train_Data.map(lambda (x, y): x).collect()

    testReview = test_Data.map(lambda (x, y): x).collect()
    testRating = test_Data.map(lambda (x, y): y).collect()
	# Send train_Data and test_Data set reviews separately for preprocessing
    trainData_Reviews = cnvrtReviews_Library(targetData_Review)
    getDec_Attributes(trainData_Reviews)
	# Create sparse matrix for the both and test_Data and train_Data dataset
    trainSparseMatrix = prep_SparseMatrix(trainData_Reviews)
    testSparseMatrix = prep_SparseMatrix(cnvrtReviews_Library(testReview))

    dataFeatures = pd.DataFrame(trainSparseMatrix, columns=dec_Attributes)
    testDataFeatures = pd.DataFrame(testSparseMatrix, columns=dec_Attributes)
	# calculate multinomial naive bayes and fit the data to the model
    nb_Model = MultinomialNB()
    model_Fitter(nb_Model)
