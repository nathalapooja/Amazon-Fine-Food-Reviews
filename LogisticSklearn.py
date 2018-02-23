import sys
import random
import math
import operator
import os
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import csv

import numpy as np  # using numpy package for mathematical operations
from numpy import *
from numpy.linalg import inv  # used for inversing the matrix
import numpy.ma

from random import randint
import math

from pyspark import SparkContext

from operator import add

import re
import string
import pandas as pd
from collections import Counter
from sklearn import linear_model

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.probability import FreqDist as nF
from nltk.corpus import stopwords
from nltk import bigrams
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize


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
    function_value.fit(dataFeatures, targetRating)
    s = f = 0
    predictedRating = list(function_value.rate_Prediction(testDataFeatures))
    for i in range(len(predictedRating)):
        if predictedRating[i] == testRating[i]:
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
stemer = PorterStemmer()
lematiser = WordNetLemmatizer()
init_System()

# Main function starts here
if __name__ == "__main__":
    sc = SparkContext(appName="Logistic regression")

    load_csvFile("Reviews.csv")# call the load_csvFile function which appends column 9 and column 6 to the new dataset named g_dataset
    trainDataFtReviews_pd = pd.DataFrame(g_dataset, columns=["review", "rating"])# constructs a dataframe from review and rating colums 
    targetData_Review = trainDataFtReviews_pd['review']# only review column is taken for further processing 

    print "Preprocessing the Data"
    trainData_Reviews = convertByRow_Reviews(trainDataFtReviews_pd)# This step calls covertByRow_Reviews method which does all data preprocessing steps

    yx_RDD = sc.parallelize(trainData_Reviews)
    train_Data, test_Data = yx_RDD.randomSplit([0.6, 0.4])
    #print trainReviews[0]

    corpus = []
    for text in train_Data.collect():
        corpus.append(text[0])
    y_train = []
    for text in train_Data.collect():
        y_train.append(text[1])

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(corpus)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    #print X_train_tfidf
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(X_train_tfidf, y_train)
    prediction = dict()
    #--- test_Data set

    print "User inputs are: "
    X_test_Data = open("input.txt").readlines()
    test_Data_set = []
    for text in X_test_Data:
        test_Data_set.append(text)
    X_new_counts = count_vect.transform(test_Data_set)
    X_test_Data_tfidf = tfidf_transformer.transform(X_new_counts)

    prediction['Logistic'] = logreg.predict(X_test_Data_tfidf)
    print prediction



    #dataFeatures = pd.DataFrame(trainSparseMatrix, columns=decisionAttributes)
    #testDataFeatures = pd.DataFrame(testSparseMatrix, columns=decisionAttributes)
	# calculate multinomial naive bayes and fit the data to the model
    #clf1 = MultinomialNB()
    #fitter(clf1)
