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

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import string

##################################

## Step 1: Read in the file
## We cannot reac it in as csv because it is a mess
## One option is to convert it to text.

### !!!! NOTICE - I am using a VERY small sample of this data
### that I created by copying the column names and first 5
### rows into a new Excel - saving as .csv - and naming it as...
#RawfileName="SAMPLE_MovieReviewsFromSYRW2.csv"

### YOU MUST CHANGE THIS PATH or place your .csv file in the same
## location(folder) as your code. 
RawfileName="C:/Users/profa/Documents/R/RStudioFolder_1/DrGExamples/R_For_DataScience/DATA/MovieDataSAMPLE_labeledVERYSMALL.csv"


FILE=open(RawfileName,"r")

## We are going to clean it and then write it back to csv!
## So, we need an empty csv file - let's make one....
filename="CleanText4.csv"
NEWFILE=open(filename,"w")
## In the first row, create a column called Label and a column Text...
ToWrite="Label,Text\n"
## Write this to new empty cs v file
NEWFILE.write(ToWrite)
## Close it up
NEWFILE.close()

    

### Now, we have an empty csv file called CLeanText.csv
### Above we created the first row of column names: Label and Text
### Next, we will open this file for "a" or append - so we can
### add things to it from where we left off
### NOTE: If you open this file again with "w" it will write over
### whatever is in the file!  USE "a"....
### This line of code opens the file for append and creates
### a variable (NEWFILE) that we can use to access and control the
### file.
NEWFILE=open(filename, "a")

### We also will build a CLEAN dataframe.
### So for now, we need a blank one...
MyFinalDF=pd.DataFrame()

###############################
## IMPORTANT
##
## Below, we will create a lot of 
## prints and outputs that we want to see
## Let's write them all to a file so
## we can see what our code is doing
###################################
OutputFile="MyOutputFile.txt"
## There are many ways to do this...
## I prefer to open the file with "w" to
## create it. Then, close and reopen with "a" to
## write to it. 
## You can also use with open, etc
OUTFILE=open(OutputFile,"w")
OUTFILE.close()
OUTFILE=open(OutputFile,"a") ### REMEMBER to close this below....


### Above, FILE is the MovieReviewsFromSYRW2.csv
### Let's go through it one row at a time....

for row in FILE:
    RawRow="The next row is: \n" + row +"\n"
    print(RawRow)
    
    OUTFILE.write(RawRow) ## I am going to write this later again for comp
    row=row.lstrip()  ## strip all spaces from the left
    row=row.rstrip()  ## strip all spaces from the right
    row=row.strip()   ## strip all extra spaces in general
    #print(row)
    ## Split up the row of text by space - TOKENIZE IT into a LIST
    Mylist=row.split(" ")
    
    #print(Mylist)
    ## Now, we will clean this list (row)
    ## We will place the results (cleaned) into a new list
    ## Therefore, we need to build a new empty list...
    NewList=[]
    
    
    for word in Mylist:
        #print("The next word is: ", word)
        PlaceInOutputFile = "The next word BEFORE is: " +  word + "\n"
        OUTFILE.write(PlaceInOutputFile)
        word=word.lower()
        word=word.lstrip()
        #word=word.strip("\n")
        #word=word.strip("\\n")
        word=word.replace(",","")
        word=word.replace(" ","")
        word=word.replace("_","")
        word=re.sub('\+', ' ',word)
        word=re.sub('.*\+\n', '',word)
        word=re.sub('zz+', ' ',word)
        word=word.replace("\t","")
        word=word.replace(".","")
        #word=word.replace("\'s","")
        word=word.strip()
        
        #word.replace("\","")
        if word not in ["", "\\", '"', "'", "*", ":", ";"]:
            if len(word) >= 3:
                if not re.search(r'\d', word): ##remove digits
                    NewList.append(word)
                    PlaceInOutputFile = "The next word AFTER is: " +  word + "\n"
                    OUTFILE.write(PlaceInOutputFile)
        
        
    print(NewList)    
    
    print(NewList[-1])  ## what is this??? Its the last element
    # What is the last element?? Its the label!

        
    label=NewList[-1]
    if "pos" in label:
        label="pos"
    else:
        label="neg"
    
    PlaceInOutputFile = "\nThe label is: " +  label + "\n"
    OUTFILE.write(PlaceInOutputFile)
    
    NewList.pop() ## removes last item
    Text=" ".join(NewList)
    
    #PlaceInOutputFile = "\nThe text  is: " +  Text + "\n"
    #OUTFILE.write(PlaceInOutputFile)
    #print(Text)
    
    #print("LABEL\n")
    #print(label)
    
    ### More cleaning....
    Text=Text.replace("\\n","")
    Text=Text.strip("\\n")
    Text=Text.replace("\\'","")
    Text=Text.replace("\\","")
    Text=Text.replace('"',"")
    Text=Text.replace("'","")
    Text=Text.replace("s'","")
    Text=Text.lstrip()
    
    #if len(Text) < 2:
     #   print("SMALL",Text)
    #print(type(Text))
    #print(Text)
    
    ## Create the string you want to write to the NEWFILE...
    OriginalRow="ORIGINAL" + RawRow
    OUTFILE.write(OriginalRow)
    
    ToWrite=label+","+Text+"\n"
   
    NEWFILE.write(ToWrite)
    
    OUTFILE.write(ToWrite)
    
    
## CLOSE files - always close files!
FILE.close()  
NEWFILE.close()
OUTFILE.close()

###########
## Read the new csv file you created into a DF or into CounterVectorizer
#######
## recall that filename is CleanFile.csv - the file we just made
## Into DF
MyTextDF=pd.read_csv(filename)
## remove any rows with NA
MyTextDF = MyTextDF.dropna(how='any',axis=0)  ## axis 0 is rowwise
#print(MyTextDF.head())
#print(MyTextDF["Label"])
#print(MyTextDF.iloc[1,1])

## KEEP THE LABELS!
MyLabel = MyTextDF["Label"]
## Remove the labels from the DF
DF_noLabel= MyTextDF.drop(["Label"], axis=1)  #axis 1 is column
#print(DF_noLabel.head())
## Create a list where each element in the list is a row from
## the file/DF
#print(DF_noLabel)
print("length: ", len(DF_noLabel))

### BUILD the LIST that "content" in CountVectorizer will expect
MyList=[]  #empty list
for i in range(0,len(DF_noLabel)):
    NextText=DF_noLabel.iloc[i,0]  ## what is this??
    ## PRINT TO FIND OUT!
    #print(MyTextDF.iloc[i,1])
    #print("Review #", i, "is: ", NextText, "\n\n")
    #print(type(NextText))
    ## This list is a collection of all the reviews. It will be HUGE
    MyList.append(NextText)

## see what this list looks like....
print(MyList[1:4])
    
########## Now we will vectorize!
## CountVectorizer takes input as content
## But- you cannot use "content" unless you know what
## this means and so what the CountVectorizer expects.
## "content" means that you will need a LIST that
## contains all the text. In other words, the first element in
## the LIST is ALL the text from review 1 (in this case)
## the second element in the LIST will be all the text from
## review 2, and so on...
## If you look ABOVE, the for loop BUILDS this LIST.
    ############################################################
MycountVect = CountVectorizer(input="content")

CV = MycountVect.fit_transform(MyList)

MyColumnNames=MycountVect.get_feature_names()
VectorizedDF_Text=pd.DataFrame(CV.toarray(),columns=MyColumnNames)
## Note - this DF starts at row 0 (not 1)
## My labels start at 1 so I need to shift by 1
print(VectorizedDF_Text)

### Put the labels back
## Make copy
print(MyLabel)
print(type(MyLabel))  

NEW_Labels = MyLabel.to_frame()   #index to 0
print(type(NEW_Labels))

NEW_Labels.index =NEW_Labels.index-1
print(NEW_Labels)

LabeledCLEAN_DF=VectorizedDF_Text
LabeledCLEAN_DF["LABEL"]=NEW_Labels
print(LabeledCLEAN_DF)


    