#############################################
## Cleaning and Preparing Text Data
## with labels
##
## csv --> dataframe using CountVectorizer
##
##
################################################
## Gates 2021
################################################

import pandas as pd
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

################################################
## Example 1: Reading and cleaning text
##            data from a csv file
##
#################################################

## Download and look at this file
# https://drive.google.com/file/d/1frni0dwDMo_3it8mRWWs78m8OBlwU0Ll/view?usp=sharing

## Here - our life is easier because the labels are clear
## and are in the first column. This will NOT always be the case

## Let's read the csv file into a DF to see what
## happens if we do.
path = "C:/Users/profa/Documents/Python Scripts/CorpusExample/"

filelocation = "RestaurantSentimentCleanerLABELEDDataSMALLSAMPLE.csv"

CSV_DF = pd.read_csv(path + filelocation)

print(CSV_DF)  ## Its a mess

## We cannot read text like this directly into a DF
## We need to clean it up first using CountVectorizer

## Recall that CountVectorizer requires specific paramters
## for example, if we had a corpus, we would use
## input = "filename" and then we would build a list of
## complete paths to our files.

## However, in this case, we do not have a corpus.
## Instead, we have a csv file that contains content such that
## each row of the csv file represents a review.

## Therefore, we will use CountVectorizer with input="content"
## Review this HERE:
## https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

## In order to use input="content" and our CountVectorizer,
## we need to BUILD a list of content.

## We need to read the text from each row in the .csv file


######### OK! Let's do this...................
##
## We need to open the file and read it one row at a time
## Let's just do that first....

My_FILE = open(path + filelocation, "r")

for next_row in My_FILE:
    print(next_row)

My_FILE.close()

## OK good!
## We now see that we can read each row of our data.
## We can also see that when we do read it, the first
## value is the LABEL (n or p in this case) followed
## by the content - the review in this case.

## So - our next goal is to read all of this into
## a list.
## To keep this correct, we can have one list for labels
## and one list for content.

My_Content_List = []
My_Labels_List = []

with open(path + filelocation, "r") as My_FILE:
    next(My_FILE)  ## skip the first row

    for next_row in My_FILE:
        # print(next_row)
        ## Let's split the label and the review.
        ## You can see that they are sperated by a ","
        Row_Elements = next_row.split(",")
        print(Row_Elements)
        print("The label is: \n", Row_Elements[0])
        print("\nThe review content is:\n", Row_Elements[1])
        ## OK! Now that we know this works, we can BUILD
        ## our lists....
        My_Content_List.append(Row_Elements[1])
        My_Labels_List.append(Row_Elements[0])

## Let's see what we built....
print(My_Content_List)
print(My_Labels_List)

## Excellent! Now we have our two lists!

## Let's now use CountVectorizer to create our dataframe

## Instantiate your CV first!
MyCV_content = CountVectorizer(input='content',
                               stop_words='english',
                               # max_features=100
                               )

## NOW - I can use it!
## When using CountVectorizer, we must use a method
## called fit_transform
## We also MUST make sure we are giving the method
## the approprate parameters.

## Recall that CountVectorizer creates a document term matrix
## I will now always call this DTM.
## Notice that the parameter we are giving here is My_Content_List
## This is our list of the content from our csv

My_DTM2 = MyCV_content.fit_transform(My_Content_List)

## BUT - we are not done! Right now, we havea DTM.
## We actually want a dataframe.
## Let's convert our DTM to a DF

## TWO Steps:
## First - use your CountVectorizer to get all the column names
ColNames = MyCV_content.get_feature_names()
print("The vocab is: ", ColNames, "\n\n")

## NEXT - Use pandas to create data frames
My_DF_content = pd.DataFrame(My_DTM2.toarray(), columns=ColNames)

## Let's look!
print(My_DF_content)

## Now - because we have labels, let's add a LABEL column to the data
## We are lucky here because the labels are already clean

print(My_Labels_List)

My_DF_content.insert(loc=0, column='LABEL', value=My_Labels_List)

## Have a look
print(My_DF_content)

## Write to csv file
My_DF_content.to_csv('MyClean_CSV_Data.csv', index=False)