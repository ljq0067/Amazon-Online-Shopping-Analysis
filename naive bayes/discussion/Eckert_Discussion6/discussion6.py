#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 16:22:11 2021

@author: haleyroberts
"""

######################################################################
# Fall, 2021
# Haley Eckert Roberts
# ANLY 501 - Discussion 6
# Naive Bayes & SVM on Text Data
######################################################################

# Libraries
import pandas as pd
import os
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
#import random as rd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
#from sklearn import preprocessing
import re  
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


##################################################
### Set up data frame
##################################################

### Create stemmer function
STEMMER=PorterStemmer()
def MY_STEMMER(str_input):
    words = re.sub(r"[^A-Za-z\-]", " ", str_input).lower().split()
    words = [STEMMER.stem(w) for w in words]
    return words

### Instantiate CountVectorizer using stemmer function
MyVect_STEM = CountVectorizer(input='filename',
                              analyzer = 'word',
                              stop_words='english',
                              tokenizer=MY_STEMMER,
                              lowercase = True
                              )

### Create empty data frame for NB
FinalDF_STEM = pd.DataFrame()

### Loop through files in corpuses to build file list and collect 
### all articles' vectorized word counts into single data frame.
for name in ["PhysicsCorpus", "HistoryCorpus"]:
    
    path = "/Users/haleyroberts/Desktop/501_Intro/Discussions/discussion6/" + name
    
    FileList=[]
    
    for item in os.listdir(path):
        
        # Append file name to file list
        next1 = path + "/" + item
        FileList.append(next1)  
        
        # Fit transform the contents of the file using CountVectorizer object
        X1 = MyVect_STEM.fit_transform(FileList)
        
        # Assign column names to variable
        ColumnNames1 = MyVect_STEM.get_feature_names()
        NumFeatures1 = len(ColumnNames1)
        
    # Create data frame from fit transformed data
    builderS = pd.DataFrame(X1.toarray(), columns=ColumnNames1)
    
    # Add column to data frame for labels
    builderS["Label"] = name
    
    # Add the data frame for document to the overall data frame
    FinalDF_STEM = FinalDF_STEM.append(builderS)
   
### Replace NaN-values with 0
FinalDF_STEM = FinalDF_STEM.fillna(0)

### Function to remove any columns with names that contain numbers
def RemoveNums(SomeDF):
    print("Running Remove Numbers function....\n")
    temp = SomeDF
    MyList = []
    for col in temp.columns:
        Logical2 = str.isalpha(col) # check for anything that is not a letter
        if(Logical2 == False):
            MyList.append(str(col))
    temp.drop(MyList, axis=1, inplace=True)
    return temp

### Call the function to remove any colunns with numbers
FinalDF_STEM = RemoveNums(FinalDF_STEM)

### View data frame
#print(FinalDF_STEM)

##################################################
### Create testing & training sets
##################################################

#rd.seed(1234)
TrainDF1, TestDF1 = train_test_split(FinalDF_STEM, test_size=0.3)
#print(TrainDF1)
#print(TestDF1)

### Separate labels from testing data set
# Assign labels to variable for use later
Test1Labels = TestDF1["Label"]
# Remove labels from testing set
TestDF1 = TestDF1.drop(["Label"], axis=1)

### Separate labels from training data set
# Assign labels to variable for use later
Train1Labels = TrainDF1["Label"]
# Remove labels from training set
TrainDF1 = TrainDF1.drop(["Label"], axis=1)

##################################################
### Naive Bayes
##################################################

### Instantiate NB
MyModelNB = MultinomialNB()

### Run NB on all training data set
NB1 = MyModelNB.fit(TrainDF1, Train1Labels)

### Use NB model to predict labels of testing set
Prediction1 = MyModelNB.predict(TestDF1)
#print(np.round(MyModelNB.predict_proba(TestDF1),2))

### View predicted labels compared to real labels
print("\nThe prediction from NB is:")
print(Prediction1)
print("\nThe actual labels are:")
print(Test1Labels)

### Create confusion matrix to view results
### (rows are true labels, columns are predicted)
topics = ['HistoryCorpus', 'PhysicsCorpus']
cnf_matrix2 = confusion_matrix(Test1Labels, Prediction1, labels = topics)
print("\nThe confusion matrix is:")
print(cnf_matrix2)

### Convert confusion matrix array to dataframe
df_confusion = pd.DataFrame(cnf_matrix2, range(2), range(2))

### Plot confusion matrix as heatmap
ax = plt.subplot()
sns.set(font_scale=1.4) # label size
sns.heatmap(df_confusion, annot=True, fmt='g', ax=ax, annot_kws={"size": 16});
ax.set_xlabel('Predicted Labels');
ax.set_ylabel('True Labels'); 
ax.set_title('Confusion Matrix - NB'); 
ax.xaxis.set_ticklabels(['History', 'Physics'])
ax.yaxis.set_ticklabels(['History', 'Physics']);
plt.show()

##################################################
### SVM
##################################################

### Instantiate SVM, linear
SVM_Model = LinearSVC(C=1)

### Fit SVM with training data
SVM_Model.fit(TrainDF1, Train1Labels)

### Create confusion matrix to view results 
SVM_matrix = confusion_matrix(Test1Labels, SVM_Model.predict(TestDF1), labels = topics)
print("\nThe confusion matrix is:")
print(SVM_matrix)
print("\n\n")

### Convert confusion matrix array to dataframe
df_confusion_SVM = pd.DataFrame(cnf_matrix2, range(2), range(2))

### Plot confusion matrix as heatmap
ax = plt.subplot()
sns.set(font_scale=1.4) # label size
sns.heatmap(df_confusion_SVM, annot=True, fmt='g', ax=ax, annot_kws={"size": 16});
ax.set_xlabel('Predicted Labels');
ax.set_ylabel('True Labels'); 
ax.set_title('Confusion Matrix - SVM'); 
ax.xaxis.set_ticklabels(['History', 'Physics'])
ax.yaxis.set_ticklabels(['History', 'Physics']);
plt.show()

### Visualize the top features for SVM
def plot_coefficients(MODEL=SVM_Model, COLNAMES=TrainDF1.columns, top_features=10):
    coef = MODEL.coef_.ravel()
    top_positive_coefficients = np.argsort(coef,axis=0)[-top_features:]
    #print(top_positive_coefficients)
    top_negative_coefficients = np.argsort(coef,axis=0)[:top_features]
    #print(top_negative_coefficients)
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    colors = ["red" if c < 0 else "blue" for c in coef[top_coefficients]]
    plt.bar(  x=  np.arange(2 * top_features)  , height=coef[top_coefficients], width=.5,  color=colors)
    feature_names = np.array(COLNAMES)
    plt.xticks(np.arange(0, (2*top_features)), feature_names[top_coefficients], rotation=60, ha="right")
    plt.show()
plot_coefficients()
