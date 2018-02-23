import sys
import random
import math
import operator
import os
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import csv

from pyspark import SparkContext

import numpy as np
from numpy import *
import numpy.ma
from numpy.linalg import inv # used for inversing the matrix

from random import randint
import math

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

from operator import add
from sklearn.model_selection import train_test_split
import scipy as sp
from operator import itemgetter
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
    #training, test = yx_RDD.randomSplit([0.6, 0.4])
    #print trainReviews[0]

    data_corpus = []
    for text in yx_RDD.collect():
        data_corpus.append(text[0])
    y_ratin = []
    for text in yx_RDD.collect():
        y_ratin.append(text[1])

    count_vect = CountVectorizer(binary='false',ngram_range=(2,2))
    data = count_vect.fit_transform(data_corpus)



    tfidf_transformer = TfidfVectorizer()
    #X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X_train_tfidf = tfidf_transformer.fit_transform(data_corpus)
    np.set_printoptions(precision=3)

    def check_ylabel(x,i):
        if(x[0]==i):
            x[0]=1
        else:
            x[0]=0
        return x

    def sigmoid(zval):
        return 1/(1+np.exp(-zval))

    def fnction_Gradient(inputVector, beta):
        y_vec=inputVector[0]
        inputVector[0]=1.0
        X = np.asmatrix(inputVector, dtype=float).T
        scores=np.dot(beta.T,X)
        residue=y_vec-sigmoid(scores)
        gradiant=np.dot(X,residue)
        return gradiant

    def rate_Prediction(X_test_Data):
        x = np.asarray(X_test_Data)
        x = np.append([1], x)
        x=x.reshape(1,x.shape[0])
        values = [(np.dot(x,w.T), class_label) for w, class_label in coef_beta_lis]
        return max(values, key=itemgetter(0))[1]




    np.set_printoptions(precision=3)

    X = X_train_tfidf
    y = np.asarray(y_ratin).astype(float)

    X_train_Data, X_test_Data, y_train, y_test = train_test_split(X, y, test_size=.4)
    y_matrix = y_train.reshape(X_train_Data.shape[0], 1)



    X_y = sp.sparse.hstack((y_matrix, X_train_Data))
    X_y.tocsr()
    Xy = sc.parallelize(X_y.tocsr().todense())

    def reduceMatrix(x):
        a = np.asarray(x)
        return a[0]

    coef_beta_lis=[]
    for i in np.unique(y):
        beta=np.zeros(X_train_Data.shape[1] + 1).reshape(X_train_Data.shape[1] + 1,1)
        Xy2 = Xy.map(lambda x: reduceMatrix(x))
        Xy_new=Xy2.map(lambda x:check_ylabel(x,i))

        num_iter=10
        alpha=0.001



        for j in range(num_iter):
            print("iteration " + str(j))
            gradiant=Xy_new.map(lambda x:fnction_Gradient(x,beta))
            gradiant = gradiant.reduce(lambda a,b:a+b)
            beta=beta+alpha*gradiant

        coef_beta_lis.append((beta.T,i))

    #rate_Prediction
    X_test_Data2 = sc.parallelize(X_test_Data.todense())


    y_pred=X_test_Data2.map(lambda x:rate_Prediction(x)).collect()
    y_pred=np.asarray(y_pred)

    #calculating accuracy
    print "=============================================================================================================="
    print('Logistic Accuracy: {0}'.format( (( y_pred == y_test.T).sum().astype(float) / len(y_test))*100) )
    print "=============================================================================================================="
