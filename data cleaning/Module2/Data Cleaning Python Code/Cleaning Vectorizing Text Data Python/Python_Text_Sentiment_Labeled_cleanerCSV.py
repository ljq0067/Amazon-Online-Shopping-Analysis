# -*- coding: utf-8 -*-
"""

@author: profa
"""

############################################################
##
## WORKING WITH TEXT IN PYTHON - CSV Files with clear labels
##
## Reading in and vectorizing
## various formats for text data
##
## This example shows what to do with 
## a very poorly formatted and dirty 
## csv file.
## 
## RestaurantSentimentCleanerLABELEDDataSMALLSAMPLE
## HERE
# https://drive.google.com/file/d/11H6AbWxKsPLY3yt__OrmK0rjjYShKhig/view?usp=sharing
#########################################
## Great tutorial
## https://kavita-ganesan.com/how-to-use-countvectorizer/#.XpIWwXJ7nb0
## Textmining Naive Bayes Example
import nltk
import pandas as pd
import sklearn
import re  
from sklearn.feature_extraction.text import CountVectorizer

#Convert a collection of raw documents to a matrix of TF-IDF features.
#Equivalent to CountVectorizer but with tf-idf norm
from sklearn.feature_extraction.text import TfidfVectorizer


from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
## For Stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import os

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import string

##################################

## In this case, our lives are easier
## because the label in this csv is 
## in the first column and is easy to find
## and to save.
##
## So, let's try an easier method for this
## cleaner csv file.
## Note however that this data still has the review
## portions in multiple columns. 
## When we read it in, these will translate into commas
## This is good - it will allow us to split by comma
## save the label, and prepare the data. 

########################################################

### !!!! NOTICE - I am using a VERY small sample of this data

### YOU MUST CHANGE THIS PATH or place your .csv file in the same
## location(folder) as your code. 
RawfileName="../DATA/RestaurantSentimentCleanerLABELEDDataSMALLSAMPLE.csv"

## This file has a header. 
## It has "setinment" and "review" on the first row.
## This will be treated like all other text and that is BAD - why?
## We need to remove it first.
## There are many ways to do this. 
## We can use seek(), we can skip it with a counter, etc. 
## But, the best way is by using "with open" and readline()

## Because the label is in the first column, we can split the
## string into two parts  so 1 split  after the first comma....
## We will create a list of labels and a list of reviews
AllReviewsList=[]
AllLabelsList=[]
#-----------------for loop---------------

with open(RawfileName,'r') as FILE:
    FILE.readline() # skip header line - skip row 1
    ## This reads the line and so does nothing with it
    for row in FILE:
        NextLabel,NextReview=row.split(",", 1)
        #print(Label)
        #print(Review)
        AllReviewsList.append(NextReview)
        AllLabelsList.append(NextLabel)
 ##----------------------------------------   
    
print(AllReviewsList)
print(AllLabelsList)

########################################
##
## CountVectorizer  and TfidfVectorizer
##
########################################
## Now we have what we need!
## We have a list of the contents (reviews)
## in the csv file.

My_CV1=CountVectorizer(input='content',
                        stop_words='english',
                        max_features=100
                        
                        )

My_TF1=TfidfVectorizer(input='content',
                        stop_words='english',
                        max_features=100
                        
                        )
## NOw I can vectorize using my list of complete paths to my files
X_CV1=My_CV1.fit_transform(AllReviewsList)
X_TF1=My_TF1.fit_transform(AllReviewsList)

print(My_CV1.vocabulary_)
print(My_TF1.vocabulary_)

## Hmm - that's not quite what we want...
## Let's get the feature names which ARE the words
## The column names are the same for TF and CV
ColNames=My_TF1.get_feature_names()

## OK good - but we want a document topic model A DTM (matrix of counts)
DataFrame_CV=pd.DataFrame(X_CV1.toarray(), columns=ColNames)
DataFrame_TF=pd.DataFrame(X_TF1.toarray(), columns=ColNames)

## Drop/remove columns not wanted
print(DataFrame_CV.columns)

## Let's build a small function that will find 
## numbers/digits and return True if so

##------------------------------------------------------
### DEFINE A FUNCTION that returns True if numbers
##  are in a string 
def Logical_Numbers_Present(anyString):
    return any(char.isdigit() for char in anyString)
##----------------------------------------------------

for nextcol in DataFrame_CV.columns:
    #print(nextcol)
    ## Remove unwanted columns
    #Result=str.isdigit(nextcol) ## Fast way to check numbers
    #print(Result)
    
    ##-------------call the function -------
    LogResult=Logical_Numbers_Present(nextcol)
    ## The above returns a logical of True or False
    
    ## The following will remove all columns that contains numbers
    if(LogResult==True):
        #print(LogResult)
        #print(nextcol)
        DataFrame_CV=DataFrame_CV.drop([nextcol], axis=1)
        DataFrame_TF=DataFrame_TF.drop([nextcol], axis=1)

    ## The following will remove any column with name
    ## of 3 or smaller - like "it" or "of" or "pre".
    ##print(len(nextcol))  ## check it first
    ## NOTE: You can also use this code to CONTROL
    ## the words in the columns. For example - you can
    ## have only words between lengths 5 and 9. 
    ## In this case, we remove columns with words <= 3.
    elif(len(str(nextcol))<=3):
        print(nextcol)
        DataFrame_CV=DataFrame_CV.drop([nextcol], axis=1)
        DataFrame_TF=DataFrame_TF.drop([nextcol], axis=1)
    
    
print(DataFrame_CV)
print(DataFrame_TF)

####################################################
##
## Adding the labels to the data......
## This would be good if you are modeling with
## supervised methods - such as NB, SVM, DT, etc.
##################################################
## Recall:
print(AllLabelsList)
print(type(AllLabelsList))

## Place these on dataframes:
## List --> DF
DataFrame_CV.insert(loc=0, column='LABEL', value=AllLabelsList)
DataFrame_TF.insert(loc=0, column='LABEL', value=AllLabelsList)
#print(DataFrame_CV)
#print(DataFrame_TF)

############################################
##
##  WRITE CLEAN, Tokenized, vectorized data
##  to new csv file. This way,  you can read it
##  into any program and work with it.
##
######################################################



CV_File="MyTextOutfile_count.csv"
TF_File="MyTextOutfile_Tfidf.csv"


######## Method 1: Advanced ---------------

## This is commented out - but you can uncomment to play with it
#import tkinter as tk
#from tkinter import filedialog
# =============================================================================
# root= tk.Tk()
# 
# def exportCSV(df):
#     #global df
#     export_file_path = filedialog.asksaveasfilename(defaultextension='.csv')
#     df.to_csv(export_file_path, index = False, header=True)
# 
# 
# USER_Button_CSV = tk.Button(text='File Saved - Close this Box', 
#                              ## Call the function here...
#                              command=exportCSV(DataFrame_CV), 
#                              bg='blue', 
#                              fg='grey', 
#                              font=('helvetica', 11, 'bold'))
# 
# USER_Button_CSV.pack(side=tk.LEFT)
# 
# root.mainloop()
# =============================================================================

##################################################

################ Method 2: Save csv directly --

DataFrame_CV.to_csv(CV_File, index = False)
DataFrame_TF.to_csv(TF_File, index = False)









