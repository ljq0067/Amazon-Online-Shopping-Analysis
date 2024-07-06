
## Gates
## Example Code
## CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split
import random as rd


#########   FOR YOU TO DO -------------------------
##
###############################################################
##
## This example set will review more advanced options for:
## reading text into a DF from two corpuses - 
## AND Labeled
## and also seperating datasets into training and testing sets
##
## Remember - you can (and should always be able to) format either
## labeled or unlabled data. 

## Also - keep in mind that there are many ways to label corpus data.
## One is to have a different corpus for each label. 
## In this example set, I will use a corpus called DOG and a corpus called HIKE
## each containing text files on that specific topic. You can so the same
## for sentiment POS, NEG, etc. and you can have more than two labels....
##
## Instead, you can also have one corpus where the file names in the corpus 
## define the label. So H1.txt and H2.txt might both be about hiking
## and D1.txt and D2.txt might be about dogs, etc. Again, you can do the same
## more many labels and for sentiment. 

## There are MANY DIFFERENT OPTIONS. I recommend trying as many as you can
## AND pushing my examples forward - meaning add/update/improve them...
##
## Also, for csv files, you can have a label in the csv or not. 
## We will also talk more about this in the lecture as I will need to show you 
## a few examples.

## For now - try to understand my code enough to CREATE the data (corpus(s) and/or csv)
## that you will need to run the code. 

## This is important! If you can build the data you need based on the code
## you really understand the code and the data formats, etc. 

## Finally, we will use CountVectorizer with two options below...as you review
## make sure that everything makes sense.

## NEVER somply cut/paste code. Always review each line, print things,
## update and add, and OWN THE CODE :)

## IMPORTANT - also create NEW FOLDERS when building corpus's

#####################################################################################
## To get started, I will instantiate two different CountVectorizers
## One will be for input="filename"  (so lists of complete file names)
## and one will be for input="content" for a list of content. 
#

# Create a CountVectorizer for you to use. This is called "instantiation"
## This CountVecotizer will be used with MyList, the list of file locations. 
My_FN_CV = CountVectorizer(input = "filename", stop_words="english")


#My_CON_CV=CountVectorizer(input = "content", stop_words="english")

## NOw I can use these as I need to.

#############################################################################
## Next, I will read in an vectorize text data from corpus POS and corpus NEG
## POS contains .txt files that are about positive things
## NEG contains .txt files that are aboout negative things
#############################################################################

## Set the path for the folder where POS and NEG are located
## This will also be the path where my csv file is.
MyPath="C:/Users/profa/Documents/Python Scripts/TextMining/Week2_3/"

## Now - ALWAYS takes the time to make sure that you can access files
## In other words - TEST you Code!

NEG_files=os.listdir(MyPath+"/NEG")
print(NEG_files)
POS_files=os.listdir(MyPath+"/POS")
print(POS_files)

## OK - good!

## Now - I want to build a LIST of complete paths to filenames - BUT - 
## This time I also want to build a LIST of labels as well.
## In other words, as I read in all the file paths from NEG,
## I also want to append "NEG" to my list of labels. 

## I will need TWO new and empty lists

MyCompleteFileNames=[]
MyLabels=[]

## Now I need a loop because there is no way to know (in real life) how
## many files are in each corpus. 
## While I can make this even more advanced and use a loop for the folder names
## as well - I will leave that to you to think about.
## I know that my two folders are called POS and NEG and so will "hard code"
## that in......................................

## You will notice that I am going to use a nested (double) loop
## The outer loop will go through each corpus. I only have two here: POS and NEG
## but this code can be updated to work for any number of corpus's 
## with some creativity........................

##_-----------------------------------------------------------
for folderName in ["POS", "NEG"]:
    ## Get the files 
    filenames=os.listdir(MyPath + folderName)
    #print(filenames)  
    
    
    for eachfile in filenames:
        #print(eachfile)
        ## The "str" assures that everything is a string
        fullpath=MyPath + str(folderName) + "/" + str(eachfile)
        # C:/Users/profa/Documents/Python Scripts/TextMining/Week2_3/POS/P1.txt
        print(fullpath)
        MyCompleteFileNames.append(fullpath)
        MyLabels.append(folderName)
##----------------------------------------------------------
        

print("THe labels are:\n")
print(MyLabels)     

print("THe complete list of file paths is:\n")
print(MyCompleteFileNames)     

########################### Next, we can use
## The proper CountVecotorizer and process the text data WITH the labels...   

## Now - let's use fit.transform to transform the list of files                                 
MyTextData1 = My_FN_CV.fit_transform(MyCompleteFileNames)
## MyTextData1 is a DTM
print(type(MyTextData1))
## Get the column names
MyCol_Names=My_FN_CV.get_feature_names()
## Be careful - if you print all the col names there may be 100s or 1000s :)

## Create the dataframe...
MyDF_Pos_Neg = pd.DataFrame(MyTextData1.toarray(), columns=MyCol_Names)
print(MyDF_Pos_Neg.head())

##_-----------------------------------
## Connect the LABELS to the DF
##---------------------------------------
print(type(MyLabels))
## It is a list. 
## Let's 
MyDF_Pos_Neg['LABEL'] = MyLabels
## Check it now
print(MyDF_Pos_Neg.head())




######### NOTICE that the data frame has numbers as columns.
## This is not good unless you are specifically looking for numeric values

## Remove all columns with numbers
print("The first DF is: \n", MyDF_Pos_Neg.head())
####### Clean-up ------------>


for nextcol in MyDF_Pos_Neg.columns:
    if(re.search(r'[^A-Za-z]+', nextcol)):
        print(nextcol)
        MyDF_Pos_Neg=MyDF_Pos_Neg.drop([nextcol], axis=1)
#    ## The following will remove any column with name
#    ## of 3 or smaller - like "it" or "of" or "pre".
#    ##print(len(nextcol))  ## check it first
#    ## NOTE: You can also use this code to CONTROL
#    ## the words in the columns. For example - you can
#    ## have only words between lengths 5 and 9. 
#    ## In this case, we remove columns with words <= 3.
    elif(len(str(nextcol))<=3):
        print(nextcol)
        MyDF_Pos_Neg=MyDF_Pos_Neg.drop([nextcol], axis=1)
        
    

        
#---------------------------------------------------------------
    
print(MyDF_Pos_Neg)

########################################################
###################  Creating Training and Testing Sets
###################  for use with supervised methods
################### Recall that supervised methods such as
###################  SVM, DT, RF, NN, NB, etc. must be 
###################  Trained on a training set and then
###################  Tested on a test set.
################### The Training Set and the Test Set
################### cannot overlap!  Why?
################### So, when we build and test models
################### We use our dataset and we SPLIT IT
################### into a Training Set and a Testing Set
#########################################################

################### Always do this AFTER all cleaning, prep
################### normalizing, stopword removal, stemming, etc
################### is done. Why?

#########################################################
################### Training Sets are a fraction of the data
################### and Testing Sets are the rest.
################### So, if the Training Set is 2/3 of the data
################### The Testing Set is the other 1/3. 
################### Is there a RULE for sizes? NOPE :) 
################### but in class, we will talk about this a bit. 
#########################################################

################### Our data is in order of label so using
################### the first 2/3 for training is a BAD idea. 
################### Why?

################### Randomly select 2/3 of the data to be the 
################### Training set - Python has package for this!

#from sklearn.model_selection import train_test_split
#import random as rd
#rd.seed(1234)  ## this will choose the same thing each time
Pos_Neg_TrainDF, Pos_Neg_TestDF = train_test_split(
        MyDF_Pos_Neg, test_size=0.3)

print("Training set\n",Pos_Neg_TrainDF)
print("Testing set\n",Pos_Neg_TestDF)

##################### Next - seperate the label from the data
## Create two data frames for the Training Set
## and two for the Testing Set so that
## one is the data (which is numeric with the words
## as the column names and the docs as the rows)  AND
## one that is just the label. 
## Why?
## Because in Python, all supervised models in sklearn 
## require as input the data  and the label for the data
## as SEPERATE dfs. 
##
## It is critical to understand that a LABEL is NOT the same
## as the data row/vector that describes that label. 

## In this case - BUT NOT IN ALL CASES :) - our
## label is called "LABEL".

## Create a DF of the just the label

## FOR TRAIN------------------------------
Train_Label_DF=pd.DataFrame(Pos_Neg_TrainDF["LABEL"])
## Check it
print(type(Train_Label_DF))
print(Train_Label_DF.head())

## Now - we need to remove the LABEL column from the 
## original dataframe so that it ONLY include the data

Pos_Neg_TrainDF=Pos_Neg_TrainDF.drop("LABEL", axis=1)  
## axis = 1 is for dropping columns. axis = 0 drops rows
print(Pos_Neg_TrainDF.head())


## FOR TEST--------------------------------------
Test_Label_DF=pd.DataFrame(Pos_Neg_TestDF["LABEL"])
## Check it
print(type(Test_Label_DF))
print(Test_Label_DF.head())

## Now - we need to remove the LABEL column from the 
## original dataframe so that it ONLY include the data

Pos_Neg_TestDF=Pos_Neg_TestDF.drop("LABEL", axis=1)  
## axis = 1 is for dropping columns. axis = 0 drops rows
print(Pos_Neg_TestDF.head())

## OK!
## Now we have our data ready to use with any ML supervised method!

##
########################## End of Clean up and prep
## Always note that you can do a lot more cleaning!


