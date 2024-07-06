#############################################
## Cleaning and Preparing Text Data
## with labels
##
## corpus --> dataframe using CountVectorizer
##
##
################################################
## Gates 2021
################################################


################################################
## Example 1: Reading and cleaning text
##            data from a corpus of .txt files
##
#################################################

## What is a corpus?

## A corpus is a folder that contains (in this case)
## text (.txt) files.
## For example, you can have a corpus of novels or
## a corpus of movie reviews or a corpus of speeches -
## etc.

################### DO THESE STEPS ...   !!!!!!!!!!!!!!!


## Let's build our own corpus for this tutorial
## First, determine where you want your new corpus
## (your new folder of files) to be. Make
## sure you know where the PATH to your corpus will be.

## Next, in that folder on your computer that you
## selected to place the corpus, create a NEW folder.

## In other words, create a new folder in a location
## on your computer.

## I will create mine here:
##  C:\Users\profa\Documents\Python Scripts\CorpusExample

## So, inside of C:\Users\profa\Documents\Python Scripts\Corpus Example
## I will create a NEW FOLDER. I will name this new folder
## MyNewCorpus.

## Recall, a corpus is a folder that contains text files.
## ALWAYS create a new folder to use as a corpus - otherwise
## you may run into errors that you will not be able to find.

## Now, inside my folder (my corpus) which is called MyNewCorpus
## I will create 6 VERY SMALL .txt files:
##  Hiking1.txt
##  Hiking2.txt
##  Pets1.txt
##  Pets2.txt
##  Cooking1.txt
##  Cooking2.txt

## Then, inside of each of these 6 .txt files, I will write (type)
## a couple of sentences about that topic. KEEP IT SIMPLE
## Do not use a lot of different words
## This is a practice example and you want it to be EASY

## Example:
## In Hiking1.txt, I have:
## I like to hike. I hike with a friend. To hike you need gear.

## In Pets1.txt, I have:
## I love my dog. My dog is my friend. My dog eats dog food.

## You get the idea?

## OK! Now - fill in all 6 .txt files with words that support
## the name of each.

#######################################################
## BEFORE YOU MOVE FORWARD.....
##
## Make sure you created a NEW corpus that has 6 (or more)
## .txt files. Name the files as above or smartly. Place some smart
## and easy text into the files.
################################################################


## In Python, we will use the sklearn CountVectorizer method
## to convert our text into a dataframe.
## This process has MANY steps.
## You can (and should) read more about CountVectorizer
## here: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

## LIBRARIES ------------------------------------------------
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd
import os
import re

import warnings

warnings.filterwarnings('ignore')

## Read the corpus files into a Document Term Matrix Object (or DTM)
path = "C:/Users/profa/Documents/Python Scripts/CorpusExample/MyNewCorpus"

## Get the text data first
print("calling os...")
FileNameList = os.listdir(path)
## check the TYPE
print(type(FileNameList))
print(FileNameList)

## Now - before we move forward, let's think about our goal.
## We have a corpus of text files.
## We want a labeled (in this case) dataframe.

## The dataframe will have each document in the corpus as a row
## each word as a column (words are variables in the text world)
## and for each document, the COUNT (frequency) of each word will
## appear.  CountVectorizer will do this for us by tokenizing the
## the text into words and then counting how many times each
## word appears in each file.

################ Let's do this BY HAND a little bit to
## make sure it makes sense....


## Next, to use CountVectorizer, we MUST understand
## HOW to use it!
## The two ways to do this are to read about it
## and to practice.

## In reading this:
## https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
## We discover (learn) that there are input options.
## In our case, because we have a corpus, we will use input="filename"
## AND we must BUILD a list of complete paths to all of our files.

## Let's do that!

## Build a new list.
## This new list will contain all complete paths to all files in
## our corpus.

## Make a blank list
MyFileNameList = []
## Also - let's same the names of the files (not the paths)
## in another list.
FileNames = []

## Now, we need to build these lists.

## Let's do this one step at a time.
## First, make sure you can access all the files....
for nextfile in os.listdir(path):
    print(nextfile)

## Next, see if you can concatentate the path to the files
for nextfile in os.listdir(path):
    fullpath = path + "/" + nextfile
    print(fullpath)

## Ok - good!
## Now - loop through, build the full paths
## and place them into your list

for nextfile in os.listdir(path):
    fullpath = path + "/" + nextfile
    # print(fullpath)
    MyFileNameList.append(fullpath)
    ## let's place the files names (not the path too)
    ## into our other list
    FileNames.append(nextfile)

## print the list....

print(MyFileNameList)
print("\\\\\\\\\\\\\\\/n")
print(FileNames)

## This is good! We now have a list of complete
## paths to all files in our corpus and we have a list
## of just their names.


###########################################################
## Using CountVectorizer to create a Document Term Matrix
##
## This will be the first step toward the dataframe....
##########################################################

## In sklearn - to use a method like CountVectorizer
## you need to first "instantiate it". This is a fancy
## way of saying that you need to make your copy....

MyCV = CountVectorizer(input='filename',
                       stop_words='english',
                       # max_features=100
                       )

## So my CountVectorizer is called MyCV

## NOW - I can use it!
## When using CountVectorizer, we must use a method
## called fit_transform
## We also MUST make sure we are giving the method
## the approprate parameters.

## Recall that CountVectorizer creates a document term matrix
## I will now always call this DTM.
## Notice that the parameter we are giving here is MyFileNameList
## This is our list of complete paths to our files in our corpus.

My_DTM = MyCV.fit_transform(MyFileNameList)

## BUT - we are not done! Right now, we havea DTM.
## We actually want a dataframe.
## Let's convert our DTM to a DF

## TWO Steps:
## First - use your CountVectorizer to get all the column names
MyColumnNames = MyCV.get_feature_names()
print("The vocab is: ", MyColumnNames, "\n\n")

## NEXT - Use pandas to create data frames
My_DF = pd.DataFrame(My_DTM.toarray(), columns=MyColumnNames)

## Notice the .toarray()
## This is CRITICAL. I need to convert my DTM to an array first
## then to the DF

## OK!

## Let's see what we have.....
print(My_DF)

## We did it!! We converted a corpus of text files into
## a dataframe.

## Now - let's add the labels.

## We do not always have labels, but in this case, we do.
## We know that we have Cooking, Hiking, and Pets.

## recall

print(FileNames)

## BUT!!! These file names are not labels YET.
## First, they are all different! Hiking1 is NOT the same
## as Hiking2, etc.

## So, our goals here are to update these names so
## that we just have three categories: Hiking, Cooking, Pets
## Then, we want to create a column in our DF called LABEL
## (or whatever) and include these labels....

## Step 1 - let's update our file names to be the categories
## we want:

## RIght now, we have:
print(FileNames)

## Make sure you can loop
for filename in FileNames:
    print(filename)

## OK! We can!

## Now - let's loop and clean


for filename in FileNames:
    # print(type(filename))
    ## remove the number and the .txt from each file name
    newName = filename.split(".")
    print(newName)
    print(newName[0])

## OK - can you see how using split split the filename
## into two parts and then we can access the first part.

## Let's now add to this and remove the number as well

CleanNames = []

for filename in FileNames:
    ## remove the number and the .txt from each file name
    newName = filename.split(".")
    # print(newName[0])
    ## remove any numbers
    newName2 = re.sub(r"[^A-Za-z\-]", "", newName[0])
    print(newName2)
    CleanNames.append(newName2)

## Let's see if we did it...
print(CleanNames)

## OK!

## Now - let's add these labels as a column to our DF

## RIght now we have
print(My_DF)

## Add the column

My_DF["LABEL"] = CleanNames

## Have a look
print(My_DF)

## Write to csv file
My_DF.to_csv('MyCleanCorpusData.csv', index=False)