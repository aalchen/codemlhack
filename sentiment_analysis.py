# -*- coding: utf-8 -*-

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Commented out IPython magic to ensure Python compatibility.
import numpy as np 
import pandas as pd 
import re
import nltk 
import matplotlib.pyplot as plt
import itertools
import csv
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.classify.util import accuracy
from sklearn.naive_bayes import BernoulliNB

features_file = "test.csv"
labels_file = "train_labels.csv"
test_file = "test.csv"

def format_sentence(sent):
    return([word for word in nltk.word_tokenize(sent)])

# Regex to clean training data 

data = []
with open(features_file) as feature, open(labels_file) as label:
    for line1, line2 in zip(feature, label):
        
        
# Remove all the special characters
processed_feature = re.sub(r'\W', ' ', line1)

# remove all single characters
processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

# Remove single characters from the start
processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 

# Substituting multiple spaces with single space
processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

# Removing prefixed 'b'
processed_feature = re.sub(r'^b\s+', '', processed_feature)

# Converting to Lowercase
processed_feature = processed_feature.lower()
        
# Negations
processed_feature = re.sub(r'no ','no_', processed_feature)
processed_feature = re.sub(r'not ','not_', processed_feature)
processed_feature = re.sub(r'never ','never_', processed_feature)
processed_feature = re.sub(r'nowhere ','nowhere_', processed_feature)
processed_feature = re.sub(r'nothing ','nothing_', processed_feature)
processed_feature = re.sub(r'hardly ','hardly_', processed_feature)
processed_feature = re.sub(r'scarcely ','scarcely', processed_feature)
processed_feature = re.sub(r'barely ','barely_', processed_feature)
processed_feature = re.sub(r'don ','don_', processed_feature)
processed_feature = re.sub(r'aren ','aren_', processed_feature)
processedfeature = re.sub(r'won','won', processed_feature)
        
        
#add to data
data.append([format_sentence(processed_feature), line2])

# Splitting into training and test data 
training = data[:int((.8)*len(data))]
test = data[int((.8)*len(data)):]

# Using Naive Bayes Classifier 
classifier = NaiveBayesClassifier.train(training)
classifier.show_most_informative_features()

print(accuracy(classifier, test))

testSet = []
with open(test_file) as test:
    for line1 in test:
        testSet.append(line1)

# Regex to clean testing data 

testSetProcessed = []
for line in testSet:

# Remove all the special characters
processed_feature = re.sub(r'\W', ' ', line)

# remove all single characters
processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

# Remove single characters from the start
processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 

# Substituting multiple spaces with single space
processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

# Removing prefixed 'b'
processed_feature = re.sub(r'^b\s+', '', processed_feature)

# Converting to Lowercase
processed_feature = processed_feature.lower()
    
# Negations
processed_feature = re.sub(r'no ','no_', processed_feature)
processed_feature = re.sub(r'not ','not_', processed_feature)
processed_feature = re.sub(r'never ','never_', processed_feature)
processed_feature = re.sub(r'nowhere ','nowhere_', processed_feature)
processed_feature = re.sub(r'nothing ','nothing_', processed_feature)
processed_feature = re.sub(r'hardly ','hardly_', processed_feature)
processed_feature = re.sub(r'scarcely ','scarcely', processed_feature)
processed_feature = re.sub(r'barely ','barely_', processed_feature)
processed_feature = re.sub(r'don ','don_', processed_feature)
processed_feature = re.sub(r'aren ','aren_', processed_feature)
processedfeature = re.sub(r'won','won', processed_feature)

#add to data
testSetProcessed.append(processed_feature)

testResults = []
for line1 in testSetProcessed:
    testResults.append(classifier.classify(format_sentence(line1)))

with open("finalresult.csv", 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerow(testResults)
