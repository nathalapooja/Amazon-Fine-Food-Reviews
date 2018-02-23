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

from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

from operator import add
from sklearn.neighbors import KNeighborsClassifier

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
        stpwrds.append(stemmer.stem(lemmatiser.lemmatize(x, pos="v")))

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

# Function is used to send each row for preprocessing
def cnvrtReviews_Library(reviews):
    convertedReviews = []
    for a in reviews:
        convertedReviews.append(string_Processing(str(a).lower()))
    return convertedReviews


# Function to prepare sparse matrix
def prep_SparseMatrix(cnvrted_Reviews):
    sparse_Matrix = []
    for cr in cnvrted_Reviews:
        newCr = [0] * len(dec_Attributes)
        for word in cr:
            if word in dec_Attributes:
                index = dec_Attributes.index(word)
                newCr[index] += 1
            else:
                pass
        sparse_Matrix.append(newCr)
    return sparse_Matrix

# To fit the model built by the scikit learn library
def model_Fitter(function_value):
    nb_functionvalue = str(function_value)
    nb_functionvalue = nb_functionvalue.partition("(")[0]
    function_value.fit(dataFeatures, tgtData_Rating)
    s = f = 0
    predictedRating = list(function_value.predict(testDataFeatures))
    for i in range(len(predictedRating)):
        if predictedRating[i] == testData_Rating[i]:
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


dataFeatures = []
testDataFeatures = []
# initializing NLTK functions for preprocessing
reg_tokens = re.compile(r'(' + '|'.join(stg_regex) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + stg_emoticons + '$', re.VERBOSE | re.IGNORECASE)
punctuation = list(string.punctuation)
stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()
init_System()

# Main function starts here
if __name__ == "__main__":
    sc = SparkContext(appName="KNN-sklearn")# creating a sparkcontext object with application name "KNN-sklearn"

    load_csvFile("Reviews.csv")#call the load_csvFile function which appends column 9 and column 6 to the new dataset named g_dataset
    trainDataFtReviews_pd = pd.DataFrame(g_dataset, columns=["review", "rating"])# constructs a dataframe from review and rating colums 
    targetData_Review = trainDataFtReviews_pd['review']# only review column is taken for further processing 

    print "Preprocesseing the Data"
    trainData_Reviews = convertByRow_Reviews(trainDataFtReviews_pd)# This step calls covertByRow_Reviews method which does all data preprocessing steps

    yx_RDD = sc.parallelize(trainData_Reviews)# An RDD named yx_RDD is created from the preprocessed data
    train_Data, test_Data = yx_RDD.randomSplit([0.6, 0.4])# Splitting thes RDD into training and test where training is used for building the model and test is used for finding the accracy of algorithm
    #test_Data = sc.parallelize(['This  is great stuff.  Made some really tasty banana bread.  Good quality and lowest price in town.'])

    tgtData_Rating = train_Data.map(lambda (x, y): y).collect()
    tgtData_Review = train_Data.map(lambda (x, y): x).collect()

    testData_Review = test_Data.map(lambda (x, y): x).collect()
    testData_Rating = test_Data.map(lambda (x, y): y).collect()

    trainReviews = cnvrtReviews_Library(tgtData_Review)
    getDec_Attributes(trainReviews)

    k = 10

    print "Preparing the KNN model"

    trainSparseMatrix = prep_SparseMatrix(trainReviews)
    testSparseMatrix = prep_SparseMatrix(cnvrtReviews_Library(testData_Review))

    dataFeatures = pd.DataFrame(trainSparseMatrix, columns=dec_Attributes)
    testDataFeatures = pd.DataFrame(testSparseMatrix, columns=dec_Attributes)

	# Calculate the K nearest neighbors, get the class based on fitting the data to the model nuilt by the library. Finally calculate the accuracy
    while (k <= 50):

        clf = KNeighborsClassifier(n_neighbors=k, weights='distance', algorithm='auto')

        clf.fit(dataFeatures, tgtData_Rating)

        print "Predicting"

        s = f = 0
        predictedRating = list(clf.predict(testDataFeatures))
        for i in range(len(predictedRating)):
            if predictedRating[i] == testData_Rating[i]:
                s += 1
            else:
                f += 1
        print "Prediction rate is for KNN with " + str(k) + " value is: " + str(float(s) / len(predictedRating) * 100.0)

        k = k + 10
