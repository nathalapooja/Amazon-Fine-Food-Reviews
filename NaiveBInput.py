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
#list of input reviews
in_Review = []


# initialize the system and removes stopwords from dataset
def init_System():
    print "Preparing stop words."
    stop_word  = set(stopwords.words('english') + punctuation + ['rt', 'via', 'i\'m', 'us', 'it'])
    for x in stop_word :
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
	
# load_InputReviews loads the input File with user reviews
def load_InputReviews(filePath):
    print "Loading the Input Reviews..."
    in_Review = open(filePath).readlines()
    print in_Review
    in_Review = sc.parallelize(in_Review)
    print in_Review

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

# Takes each word and calculate the probability of occuring of that word in the class by taking the log as well for underfolow prevention
def get_Probability(words):
    

    problty_1 = 0.0 + math.log10(classProbability.get("1", 1.0))
    problty_2 = 0.0 + math.log10(classProbability.get("2", 1.0))
    problty_3 = 0.0 + math.log10(classProbability.get("3", 1.0))
    problty_4 = 0.0 + math.log10(classProbability.get("4", 1.0))
    problty_5 = 0.0 + math.log10(classProbability.get("5", 1.0))

    final_Probability = 0

    for i in range(len(words)):
        if (one_Words.has_key(words[i])):
            problty_1 = problty_1 + math.log10(one_Words[words[i]])

        if (two_Words.has_key(words[i])):
            problty_2 = problty_2 + math.log10(two_Words[words[i]])

        if (three_Words.has_key(words[i])):
            problty_3 = problty_3 + math.log10(three_Words[words[i]])

        if (four_Words.has_key(words[i])):
            problty_4 = problty_4 + math.log10(four_Words[words[i]])

        if (five_Words.has_key(words[i])):
            problty_5 = problty_5 + math.log10(five_Words[words[i]])

    max_Probability = problty_1
    final_Probability = 1

    if (problty_1 == 0.0 and problty_2 == 0.0 and problty_3 == 0.0 and problty_4 == 0.0 and problty_5 == 0.0):
        final_Probability = 0
    else:
        if (problty_2 > max_Probability):
            max_Probability = problty_2
            final_Probability = 2
        if (problty_3 > max_Probability):
            max_Probability = problty_3
            final_Probability = 3
        if (problty_4 > max_Probability):
            max_Probability = problty_4
            final_Probability = 4
        if (problty_5 > max_Probability):
            max_Probability = problty_5
            final_Probability = 5

    return final_Probability


# this function is used to calculate the
def get_RvwAccuracy(predictions):
    correct = 0
    for x in range(len(predictions)):
        if predictions[x][0] == predictions[x][1]:
            correct += 1
    return (correct / float(len(predictions))) * 100.0

# Other required global variables
ratinglist = []
len_Vocablry = 0
len_One = 0
len_Two = 0
len_Three = 0
len_Four = 0
len_Five = 0
n = 0
one_Words = {}
two_Words = {}
three_Words = {}
four_Words = {}
five_Words = {}
classProbability = {}

# initializing NLTK functions for preprocessing
reg_tokens = re.compile(r'(' + '|'.join(stg_regex) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + stg_emoticons + '$', re.VERBOSE | re.IGNORECASE)
punctuation = list(string.punctuation)
stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()
init_System()


# Main function start here
if __name__ == "__main__":

    sc = SparkContext(appName="NBUserInput")# creating a sparkcontext object with application name "NBUserInput"

    load_csvFile("Reviews.csv")# call the load_csvFile function which appends column 9 and column 6 to the new dataset named g_dataset
    load_InputReviews("input.txt")
    trainDataFtReviews_pd = pd.DataFrame(g_dataset, columns=["review", "rating"])# constructs a dataframe from review and rating colums 
    targetData_Review = trainDataFtReviews_pd['review']# only review column is taken for further processing 
	
    print "Preprocessing the Data"
    trainData_Reviews = convertByRow_Reviews(trainDataFtReviews_pd)# This step calls covertByRow_Reviews method which does all data preprocessing steps

    print "Building the NaiveBayes Model"

    train_Data = sc.parallelize(trainData_Reviews)# An RDD is created from the preprocessed data

    wordsinTrainData1 = train_Data.map(lambda x: tuple(x)).map(lambda x: x[0].split(' '))#converting each preprocessed review into word tuples(removes duplicate words)
    wordsinTrainData2 = [y for x in wordsinTrainData1.collect() for y in x]# creates a list of words from each tuple
    wordsinTrainData = Counter(wordsinTrainData2)# Counter is a container that keeps track of how many times equivalent values are added
    len_Vocablry = len(wordsinTrainData)# Returns the length of the argument

    n = train_Data.count()#number of reviews in training data

    ratingreviewlist = train_Data.map(lambda (x, y): (y[-1], x)).groupByKey().map(lambda x: (x[0], tuple(x[1]))).persist()# We are persisting in memory the reviews whivh are grouped by keys as ratings
    # Calculating number of words in each class
    one_Words1 = sc.parallelize(ratingreviewlist.lookup('1')).flatMap(lambda xs: [(x, 1) for x in xs]).map(
        lambda x: x[0].split(' '))# An RDD is generated from the [tuple,1] from ratingreviewlist with key value as 1
    one_Words2 = [y for x in one_Words1.collect() for y in x]# creates a list of words which belongs to rating group 1 from each tuple
    one_Words = Counter(one_Words2)# Counter is a container that keeps track of how many times equivalent values are added
    len_One = len(one_Words)

    two_Words1 = sc.parallelize(ratingreviewlist.lookup('2')).flatMap(lambda xs: [(x, 1) for x in xs]).map(
        lambda x: x[0].split(' '))# An RDD is generated from the [tuple,1] from ratingreviewlist with key value as 2
    two_Words2 = [y for x in two_Words1.collect() for y in x]# creates a list of words which belongs to rating group 2 from each tuple
    two_Words = Counter(two_Words2)# Counter is a container that keeps track of how many times equivalent values are added
    len_Two = len(two_Words)

    three_Words1 = sc.parallelize(ratingreviewlist.lookup('3')).flatMap(lambda xs: [(x, 1) for x in xs]).map(
        lambda x: x[0].split(' '))# An RDD is generated from the [tuple,1] from ratingreviewlist with key value as 3
    three_Words2 = [y for x in three_Words1.collect() for y in x]# creates a list of words which belongs to rating group 3 from each tuple
    three_Words = Counter(three_Words2)# Counter is a container that keeps track of how many times equivalent values are added
    len_Three = len(three_Words)

    four_Words1 = sc.parallelize(ratingreviewlist.lookup('4')).flatMap(lambda xs: [(x, 1) for x in xs]).map(
        lambda x: x[0].split(' '))# An RDD is generated from the [tuple,1] from ratingreviewlist with key value as 4
    four_Words2 = [y for x in four_Words1.collect() for y in x]# creates a list of words which belongs to rating group 4 from each tuple
    four_Words = Counter(four_Words2)# Counter is a container that keeps track of how many times equivalent values are added
    len_Four = len(four_Words)

    five_Words1 = sc.parallelize(ratingreviewlist.lookup('1')).flatMap(lambda xs: [(x, 1) for x in xs]).map(
        lambda x: x[0].split(' '))# An RDD is generated from the [tuple,1] from ratingreviewlist with key value as 5
    five_Words2 = [y for x in five_Words1.collect() for y in x]# creates a list of words which belongs to rating group 5 from each tuple
    five_Words = Counter(five_Words2)# Counter is a container that keeps track of how many times equivalent values are added
    len_Five = len(five_Words)
	
    alpha = 2.0
    alpha_Vocablry = alpha * len_Vocablry

    one_Words = dict(sc.parallelize(one_Words.items()).map(
            lambda (x, y): (x, np.divide(np.add(y, alpha), np.add(len_One, alpha_Vocablry)))).collect())

    two_Words = dict(sc.parallelize(two_Words.items()).map(
            lambda (x, y): (x, np.divide(np.add(y, alpha), np.add(len_Two, alpha_Vocablry)))).collect())

    three_Words = dict(sc.parallelize(three_Words.items()).map(
            lambda (x, y): (x, np.divide(np.add(y, alpha), np.add(len_Three, alpha_Vocablry)))).collect())

    four_Words = dict(sc.parallelize(four_Words.items()).map(
            lambda (x, y): (x, np.divide(np.add(y, alpha), np.add(len_Four, alpha_Vocablry)))).collect())

    five_Words = dict(sc.parallelize(five_Words.items()).map(
            lambda (x, y): (x, np.divide(np.add(y, alpha), np.add(len_Five, alpha_Vocablry)))).collect())

    classProbability = dict(ratingreviewlist.map(lambda x: (x[0], len([item for item in x[-1] if item]))).map(lambda (x, y): (x, (y * 1.0) / n)).collect())

    print "User inputs are: "
    print in_Review
    in_Review = open("input.txt").readlines()
    print in_Review
    outpu = open('output.txt', 'w')
    i = 0
    outpu.write("id \t\t review  \t\t rating \n")

    # Fit user input data to the model and predict the rating
    for inpeach in in_Review:
        userReview = [inpeach.strip()]
        userReviews = convertReviewsUser(userReview)
        test_Data = sc.parallelize(userReviews)
        testWords1 = test_Data.map(lambda x: tuple(x)).map(lambda x: get_Probability(x))
        predictions = testWords1.collect()
        #The predicted Rating from NaiveBayes for the given review is"
        #print predictions
        i = i + 1
        outpu.write(repr(i) + "  " + repr(userReview[0]) + "  " + repr(predictions[0]) + "\n")
    outpu.close()
