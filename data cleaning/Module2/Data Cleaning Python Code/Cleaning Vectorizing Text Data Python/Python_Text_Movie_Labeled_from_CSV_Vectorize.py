# -*- coding: utf-8 -*-
"""


@author: profa
"""

############################################################
##
## WORKING WITH TEXT IN PYTHON - CSV Files
##
## Reading in and vectorizing
## various formats for text data
##
## This example shows what to do with 
## a very poorly formatted and dirty 
## csv file.
## I will use the MovieReviews csv file. 
## Here is a link to the raw and original
## file: MovieReviewsFromSYRW2.csv which is HERE...
## https://drive.google.com/file/d/1KgycYN1G4zU9IHscZWDTiAn7j-qg-aIz/view?usp=sharing
#########################################

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
import os.path

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import string

########################################################
##
##  GOAL 1: Read very dirty and labeled 
##          csv file (each row is a movie review
##          the labels are pos and neg - 
##          GET this data into a COPRUS and then
##          Use CountVectorizer to process it.
########################################################

## Step 1: Read in the file
## We cannot reac it in as csv because it is a mess
## One option is to convert it to text.

### !!!! NOTICE - I am using a VERY small sample of this data
### that I created by copying the column names and first 5
### rows into a new Excel - saving as .csv - and naming it as...
#RawfileName="SAMPLE_MovieReviewsFromSYRW2.csv"

### YOU MUST CHANGE THIS PATH or place your .csv file in the same
## location(folder) as your code. 
RawfileName="C:/Users/profa/Documents/Python Scripts/ANLY503/DATA/MovieDataSAMPLE_labeledVERYSMALL.csv"
FILE=open(RawfileName,"r")  ## Don't forget to close this!


#################  Create a new Corpus Folder using os------
path="C:/Users/profa/Documents/Python Scripts/ANLY503/DATA/SmallMovieCorpus"
## You can only do this one!
IsFolderThere=os.path.isdir(path)
print(IsFolderThere)
print(not(IsFolderThere))

if (not(IsFolderThere)):
    MyNewCorpus=os.makedirs(path)

## Next, use a loop to read each row (line) in the csv file
## and write everything to a .txt file that will be saved to
## your new corpus.

## Start by checking that everything is working...
## Make sure you are at the start/beginning of the file
FILE.seek(0)

for row in FILE:
    RawRow="The next row is: \n" + row +"\n"
    #print(RawRow)
    
## OK - this works....so now we can loop through and create 
## .txt files from each row....

FILE.seek(0)
counter=-1
for row in FILE:
    RawRow="The next row is: \n" + row +"\n"
    print(RawRow)
    
    ## Next, and this WILL BE DIFFERENT FOR DIFFERENT DATA
    ## Remember - all data is unique - all cleaning and prep
    ## methods are individual. THIS IS NOT A BLACK BOX!
    
    ## In this case, the LABEL, you will notice, is at the END
    ##  of each row. 
    ## To get to the Label and use it to name each .txt file in the
    ## corpus, we will use SPLIT. In Python, using split on a string
    ## creates a list.
    
    ## First - before splitting - let's remove (strip) all
    ## the newlines, etc. 
    NextRow=row.strip()
    NextRow=NextRow.strip("\n")
    NextRow=NextRow.rstrip(",") ## right strip
    
    MyList=NextRow.split(",")
    print(MyList)
    
    ## Now - this list contains a lot of blanks...
    ## Blanks look like ''  or "". We want to remove them.
    ## Here is an advanced method in Python to remove blanks
    ## (or anything else) from a list
    
    ###########     NOTE     ##############
    ## Lambdas are small anonymous functions 
    ## They are restricted functions which do not need a name
    ################################################################
    My_Blank_Filter = filter(lambda x: x != '', MyList)
    ## Convert it back to a list - but without the blanks
    MyList = list(My_Blank_Filter)
    
    ## Let's look:
    #print("\n\nNEXT LIST \n:", MyList)
    
    ## Now - why did we need to do all of the above???
    ## The best way to answer this is to remove the times and 
    ## add them back in parts to see what things
    ## look like. 
    ## HOWEVER - we did this so that the VERY LAST ITEM
    ## in each list IS THE LABEL. This is HUGE. Now we can find it
    ## and we can use it.
    
    TheLabel=MyList[-1]
    print(TheLabel)
    MyList.pop()
    ## Also - let's REMOVE this label from the list
    ## otherwise the label will be in the data
    ## and that is NOT PERMITTED when training and testing
    ## models
    
    
    ## When you look at the output by uncommenting the print
    ## statements above - you will see that the first list is
    ## NEXT LIST 
    ## : ['text', 'reviewclass']
    ## 
    ## This is because this is the first row in the csv file.
    ## We DO NOT want to use this when we build the corpus of
    ## .txt files - so we will use a trick to skip it.
    ## That is why we have a counter.
    
    counter=counter+1   ## this let's us know that we went
                ## through the loop at least once.
                
    if(counter>=1):
        ## If we are NOT in the first loop then we can
        ## start to build the .txt files.
        NewFileName=TheLabel+str(counter)+".txt"
        print(NewFileName)
        
        NewFilePath=path+"/"+NewFileName
        n_file=open(NewFilePath, "w")
        
        ## Now, we want to write the contents of MyList
        ## into a file. BUT - not as a list.
        
        MyNewString=" ".join(MyList)
        n_file.write(MyNewString)
        n_file.close()
    
    
FILE.close()

#################################################
##
## Now we have created a corpus and the names
## of the files in the corpus are the sentiment labels
## From here - we can use CountVectorizer to 
## Process this corpus into a DF
##
#############################################################

## Recall: This is where MY corpus is and what it is called
# C:\Users\profa\Documents\Python Scripts\ANLY503\DATA\SmallMovieCorpus
    
## From above, the variable "path" is the path to our new corpus

print(path)
print(os.listdir(path))

## We will use CountVectorizer to format the corpus into a DF...
###------------------------------------------------------
######## First - build the list of complete file paths
###--------------------------------------------------------

ListOfCompleteFilePathsMovies=[]
ListOfJustFileNamesMovies=[]

for filename in os.listdir(path):
    
    print(path+ "/" + filename)
    next=path+ "/" + filename
    
    next_file_name=filename.split(".")   
    nextname=next_file_name[0]
    ListOfCompleteFilePathsMovies.append(next)
    ListOfJustFileNamesMovies.append(nextname)

#print("DONE...")
print("full list...")
print(ListOfCompleteFilePathsMovies)
print(ListOfJustFileNamesMovies)

############# Now - use CountVectorizer.....................

MyVect5=CountVectorizer(input='filename',
                        stop_words='english',
                        max_features=100
                        )
## NOw I can vectorize using my list of complete paths to my files
X_Movies=MyVect5.fit_transform(ListOfCompleteFilePathsMovies)


## Hmm - that's not quite what we want...
## Let's get the feature names which ARE the words
ColumnNamesMovies=MyVect5.get_feature_names()

## OK good - but we want a document topic model A DTM (matrix of counts)
CorpusDF_Movies=pd.DataFrame(X_Movies.toarray(),
                              columns=ColumnNamesMovies)
print(CorpusDF_Movies)

## Now update the row names
MyDictMovies={}
for i in range(0, len(ListOfJustFileNamesMovies)):
    MyDictMovies[i] = ListOfJustFileNamesMovies[i]

print("MY DICT:", MyDictMovies)
        
CorpusDF_Movies=CorpusDF_Movies.rename(MyDictMovies, axis="index")
print(CorpusDF_Movies)

############################################################
###
###            Using TfidfVectorizer
###
#############################################################

############# Now - use CountVectorizer.....................

My_TF_Vect5=TfidfVectorizer(input='filename',
                        stop_words='english',
                        max_features=100
                        )
## NOw I can vectorize using my list of complete paths to my files
X_Movies_TF=My_TF_Vect5.fit_transform(ListOfCompleteFilePathsMovies)


## Hmm - that's not quite what we want...
## Let's get the feature names which ARE the words
ColumnNamesMoviesTF=My_TF_Vect5.get_feature_names()

## OK good - but we want a document topic model A DTM (matrix of counts)
CorpusDF_MoviesTF=pd.DataFrame(X_Movies_TF.toarray(),
                              columns=ColumnNamesMoviesTF)
print(CorpusDF_MoviesTF)

## Now update the row names
MyDictMovies={}
for i in range(0, len(ListOfJustFileNamesMovies)):
    MyDictMovies[i] = ListOfJustFileNamesMovies[i]

print("MY DICT:", MyDictMovies)
        
CorpusDF_MoviesTF=CorpusDF_MoviesTF.rename(MyDictMovies, axis="index")
print(CorpusDF_MoviesTF)

## Row names
print(CorpusDF_MoviesTF.index)




##########################################
##
## Labeled data
##
## So far, we do have row names. However, 
## We do not quite have labeled data.
## Why not?
## How can we create Training and Testing 
## data for supervised methods?
##############################################



    


