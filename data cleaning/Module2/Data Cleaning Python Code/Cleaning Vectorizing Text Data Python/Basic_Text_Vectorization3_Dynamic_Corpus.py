
"""
@author: profa
"""
####################################################
###             WORKING WITH TEXT IN PYTHON
###
###             Reading in a Corpus 
###             
### This code is an example of 
### tokenization
### vectorization
### Dealing with a corpus
###  k means
### Dealing with a labeled CSV
### DIstance measures
### Frequency
### Normalization
### Formats
###
### Gates
###
### !!!!! YOU will need to CREATE things to use this code
###
### 1) Create a folder (corpus) of text files. 
###    Make text files very short and very topic 
###    specific. I have a few .txt files on dogs, and a few on hiking

### IMPORTANT:  Create a corpus so that you have text files like
###             Dog1.txt, Dog2.txt... and then Hike1.txt, Hike2.txt, etc.
###    
####################################################
import nltk
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
#Convert a collection of raw documents to a matrix of TF-IDF features.
#Equivalent to CountVectorizer but with tf-idf norm
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
## For Stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import os


##  Using input = "filename" with CountVectorizer requires 
## a LIST of complete paths (or relative but complete is better)
## to all text files.  LET'S BUILD ONE.....

## Here is MY PATH - update this to YOUR PATH TO YOUR CORPUS
path="C:/Users/profa/Documents/Python Scripts/ANLY503/DATA/CorpusHikeDog_Small"
print("calling os...")

## The "os" library in python will allow you to access your operating
## system so that you can access folders (called directories)
## Use PRINT to test it to see what it does. Always do this with new things.
print(os.listdir(path))

## Now that you see that it works - save the list of files 
## Save the list
FileNameList=os.listdir(path)
## check the TYPE
print(type(FileNameList))
print(FileNameList)

## OK good - now I have a list of the filenames

## While one would think that this would just work directly - it will not!!
## WHen using CountVecotrizer with input="filename", you need a LIST of the 
## COMPLETE PATHS to each file.
## So now that I have the list of file names in my corpus, I will build
## a list of complete paths to each....

## I need an empty list to start with:
## Notice that I defined path above.
ListOfCompleteFilePaths=[]    #empty list that will hold (eventually) all complete
## file paths. This is what I will use with CountVectorizer.

## Here - I am also creating another empty list where I will store the file
## names so I can use them to CREATE LABELS for my data.
ListOfJustFileNames=[]


########################################
## Here, we loop through all files in the
## path. For each file, we BUILD a complete
## path and place that complete path into
## The list we created above. At the same
## time we will also store the file names
## (just the names) so we can use them
## later as labels....
######################################################
for name in os.listdir(path):
    #C:\\Users\\profa\\Documents\\Python Scripts\\TextMining\\Week2_3\\CorpusHikeDog_Small\\Dog1.txt
    print(path+ "/" + name)  ## concatenate your path to \\ to the file name.
    ## Print the above to SEE what it does. 
    nextfile=path+ "/" + name   ## save the complete path the you just built
    ListOfCompleteFilePaths.append(nextfile)  ## place it in the list.
    
    #### Here - we are taking name and splitting it with .
    #### This will give us JUST the file name and not the .txt part
    nextnameL=name.split(".")   ##If name is Dog1.txt is splits it into Dog1   and txt
    print(nextnameL[0])  ### SEE what this is. Why I am using [0]? Make sure you know!
    ListOfJustFileNames.append(nextnameL[0]) ## add the name to the list of names

###########################-------------------end of for loop ------------------
    
## Let's SEE what we created................
print("List of complete file paths...\n")
print(ListOfCompleteFilePaths)
print("list of just names:\n")
print(ListOfJustFileNames)


#########################################################################
## Now we can use CountVectorizer.............................
## CountVectorizers must be set as 'content', 'file', or 'filename'
        #If set as ‘filename’, the **sequence passed as an argument to fit**
        #is expected to be a list of filenames 
        #https://scikit-learn.org/stable/modules/generated/
        ##sklearn.feature_extraction.text.CountVectorizer.html#
        ##examples-using-sklearn-feature-extraction-text-countvectorizer

        ## We will use input="filename" because we have a list
        ## of complete paths to filenames. This is what we just created
        ## above.
##################################################################
###
###             Use CountVectorizer to convert text
###             to a document-term matrix and then a Dataframe
###
#####################################################################

## Step 1: Instantiate YOUR own CountVectorizer. Mine here is called MyVect3
MyVect3=CountVectorizer(input='filename', ##notice input="filename
                        stop_words='english',
                        #max_features=1000  ## this is an option
                        )

## NOw I can vectorize using my list of complete paths to my files
## Using MyVect3 I can use the fit_transform method. This method
## takes as a parameter the list of complete file paths that we created above.
## This is why we needed to create the list of file paths.
X_DH=MyVect3.fit_transform(ListOfCompleteFilePaths)
####################################
## NOw - what do we have?
## Let's SEE
## Note that X_DH is a variable name. 
## What I have created here is a DocumentTermMatrix
#################################
print(X_DH)   ## NOt pretty to look at! Let's REFORMAT IT to a Dataframe....

## Hmm - that's not quite what we want...
## Let's get the feature names which ARE the words
## The method  get_feature_names  will return all the WORDS in all the docs
## These are called the "features" because they are the names of the columns. 
ColumnNames3=MyVect3.get_feature_names()
print(ColumnNames3)  ## have a look!

## OK good - NOW convert to a DataFrame. 
## !!! NOTICE that I first had to convert my X_DH to an array. If you skip
## this, the code will not work. 

CorpusDF_DogHike=pd.DataFrame(X_DH.toarray(),columns=ColumnNames3)
print(CorpusDF_DogHike)   ## Here is our dataframe!

###########################################################
##
##              Now we need to add the labels to 
##              our dataframe. This will take a few
##              steps....
############################################################

## First, let's create an empty dictionary. 
## A dictionary is a very useful structure in python. 
MyDict={}

## Next, let's use a loop from 0 to how ever many files
## we have. Notice that we use len here and not a number.
## This means that we can have any number of files in our
## Corpus and this will still work. That is important!
## Also, unlike R, Python starts at 0 and not 1 for most things.

for i in range(0, len(ListOfJustFileNames)):
    MyDict[i] = ListOfJustFileNames[i]  ## build the dictionary
    ## The above will associate MyDict[0] with the first name
    ## and MyDict[1] with the second name, and so on....
##------------------------end of for loop---------------------

## Let's SEE what we created....
print("MY DICT:", MyDict)

### This is NOT good enough!
### We need the labels to be categories. 
### Dog1 and Dog2, etc are NOT THE SAME to python.
## Let's fix this...

## Let's test the idea first
temp = MyDict[0].rstrip('0123456789')
print(temp)
## Good! It works!!

## Now - let's FIX the entire dictionary
for i in range(0, len(ListOfJustFileNames)):
    MyDict[i] = MyDict[i].rstrip('0123456789')
    ## The [0] will give us just the name and not the number.


### CHECK IT!
print("MY DICT:", MyDict)  
## OK - now we can place this into our dataframe....    
CorpusDF_DogHike=CorpusDF_DogHike.rename(MyDict, axis="index")
print(CorpusDF_DogHike)
## That's pretty!
## WHat can you do from here? Anything!
## First - see what data object type you actually have
print(type(CorpusDF_DogHike))

#####################################################
## 
##        More advanced code is below....
##
########################################################
## We have what we expected - a data frame.

# Convert DataFrame to matrix
MyMatrixDogHike = CorpusDF_DogHike.values
## Check it

print(type(MyMatrixDogHike))
print(MyMatrixDogHike)






##################################################################
###
###             Use TfidfVectorizer to convert text
###             to a document-term matrix and then a Dataframe
###             and normalize using tf-idf
###
#####################################################################
MyVect_TF=TfidfVectorizer(input='filename',
                        stop_words='english',
                        max_features=100
                        )
## NOw I can vectorize using my list of complete paths to my files
TF_DH=MyVect_TF.fit_transform(ListOfCompleteFilePaths)

## NOw - what do we have?
##print(X_DH)

## Hmm - that's not quite what we want...
## Let's get the feature names which ARE the words
ColumnNamesTF=MyVect_TF.get_feature_names()
#print(ColumnNames3)

## OK good - but we want a document topic model A DTM (matrix of counts)
CorpusDF_DogHike_TF=pd.DataFrame(TF_DH.toarray(),columns=ColumnNamesTF)
print(CorpusDF_DogHike_TF)

## Now update the row names
MyDict={}
for i in range(0, len(ListOfJustFileNames)):
    MyDict[i] = ListOfJustFileNames[i]

print("MY DICT:", MyDict)
        
CorpusDF_DogHike_TF=CorpusDF_DogHike_TF.rename(MyDict, axis="index")
print(CorpusDF_DogHike_TF)
## That's pretty!
## WHat can you do from here? Anything!
## First - see what data object type you actually have


print(type(CorpusDF_DogHike_TF))


## We have what we expected - a data frame.

# Convert DataFrame to matrix
MyMatrixDogHike = CorpusDF_DogHike.values
## Check it

print(type(MyMatrixDogHike))
print(MyMatrixDogHike)


#######################################################
###
###               k means clustering
###
#######################################################
# Using sklearn
## you will need
## from sklearn.cluster import KMeans
## import numpy as np
kmeans_object = sklearn.cluster.KMeans(n_clusters=2)
#print(kmeans_object)
kmeans_object.fit(MyMatrixDogHike)
# Get cluster assignment labels
labels = kmeans_object.labels_
#print(labels)
# Format results as a DataFrame
Myresults = pd.DataFrame([CorpusDF_DogHike.index,labels]).T
print(Myresults)

### Hmmm -  these are not great results
## This is because my dataset if not clean
## I still have stopwords
## I still have useless or small words < size 3

## Let's clean it up....
## Let's start with this: 

print(CorpusDF_DogHike)


###############################################
###
###         From here and down - I am doing
###         things by hand so you can see how
###         to do things without packages
###
#####################################################
## Let's remove our own stopwords that WE create

## Let's also remove all words of size 2 or smaller
## Finally, without using a stem package - 
## Let's combine columns with dog, dogs
## and with hike, hikes, hiking

## We need to "build" this in steps.
## First, I know that I need to be able to access
## the columns...

for name in ColumnNames3:
    print(name)

## OK - that works...
## Now access the column by name

for name in ColumnNames3:
    print(CorpusDF_DogHike[name])

## OK - can we "add" columns??
## lets test some things first
    
 
name1="hikes"
name2="hike"
if(name1 == name2):
    print("TRUE")
else:
    print("FALSE")

name1=name1.rstrip("s")
print(name1)
if(name1 == name2):
    print("TRUE")
else:
    print("FALSE")
    
    #############################

## RE: https://docs.python.org/2.0/lib/module-string.html
## Now - let's put these ideas together
    ## note that strip() takes off the front
    ## rstrip() takes off the rear
## BEFORE

   

############################################
### Had a very odd 3-hour issue
### My code would not remove or see "and"
### Then I noticed that my for loop was
### "skipping" items that I thought were there
### WHy is this true?
### SOlution - always make a copy
### do not "do work" as you iterate - it
### messes up the index behind the scenes

###########################################
print("The initial column names:\n", ColumnNames3)
print(type(ColumnNames3))  ## This is a list
MyStops=["also", "and", "are", "you", "of", "let", "not", "the", "for", "why", "there", "one", "which"]   

 ## MAKE COPIES!
CleanDF=CorpusDF_DogHike
print("START\n",CleanDF)
## Build a new columns list
ColNames=[]

for name in ColumnNames3:
    #print("FFFFFFFF",name)
    if ((name in MyStops) or (len(name)<3)):
        #print("Dropping: ", name)
        CleanDF=CleanDF.drop([name], axis=1)
        #print(CleanDF)
    else:
        ## I MUST add these new ColNames
        ColNames.append(name)
        

#print("END\n",CleanDF)             
print("The ending column names:\n", ColNames)


for name1 in ColNames:
    for name2 in ColNames:
        if(name1 == name2):
            print("skip")
        elif(name1.rstrip("e") in name2):  ## this is good for plurals
            ## like dog and dogs, but not for hike and hiking
            ## so I will strip an "e" if there is one...
            print("combining: ", name1, name2)
            print(CorpusDF_DogHike[name1])
            print(CorpusDF_DogHike[name2])
            print(CorpusDF_DogHike[name1] + CorpusDF_DogHike[name2])
            
            ## Think about how to test this!
            ## at first, you can do this:
            ## NEW=name1+name2
            ## CleanDF[NEW]=CleanDF[name1] + CleanDF[name2]
            ## Then, before dropping any columns - print
            ## the columns and their sum to check it. 
            
            CleanDF[name1] = CleanDF[name1] + CleanDF[name2]
            
            ### Later and once everything is tested - you
            ## will include this next line of code. 
            ## While I tested everyting, I had this commented out
            ###   "******
            CleanDF=CleanDF.drop([name2], axis=1)
        
print(CleanDF.columns.values)

## Confirm that your column summing is working!

print(CleanDF["dog"])
#print(CleanDF["dogs"])
#print(CleanDF["dogdogs"])  ## this should be the sum

## AFTER
print(CleanDF)

## NOW - let's try k means again....
############################## k means ########################

# Convert DataFrame to matrix
MyMatrixClean = CleanDF.values
## Check it
print(type(MyMatrixClean))
print(MyMatrixClean)

# Using sklearn
## you will need
## from sklearn.cluster import KMeans
## import numpy as np
kmeans_object2 = sklearn.cluster.KMeans(n_clusters=3)
#print(kmeans_object)
kmeans_object2.fit(MyMatrixClean)
# Get cluster assignment labels
labels2 = kmeans_object2.labels_
print("k-means with k = 3\n", labels2)
# Format results as a DataFrame
Myresults2 = pd.DataFrame([CleanDF.index,labels2]).T
print("k means RESULTS\n", Myresults2)

################# k means with k = 2 #####################


# Using sklearn
## you will need
## from sklearn.cluster import KMeans
## import numpy as np
kmeans_object3 = sklearn.cluster.KMeans(n_clusters=2)
#print(kmeans_object)
kmeans_object3.fit(MyMatrixClean)
# Get cluster assignment labels
labels3 = kmeans_object3.labels_
print("K means with k = 2\n", labels3)
# Format results as a DataFrame
Myresults3 = pd.DataFrame([CleanDF.index,labels3]).T
print("k means RESULTS\n", Myresults3)







