
## Gates
## Example Code
## CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import os
#########   FOR YOU TO DO -------------------------
##
## Create a NEW folder. Inside the folder, create three SMALL .txt documents.
## Name the .txt documents  T1.txt, T2.txt, T3.txt
## Keep your files VERY simple and short.
## 
## Make sure you know the path (location on your computer) where the files are.
## Here is an example of where they are on MY computer. This WILL NOT work for you
## because your path names are not the same as mine :)                              
##     Update the following to reflect YOUR file locations.....                   
##
## Here, we have created a LIST of complete paths to the files in our courpus. 
## We do this when we use CountVectorizer with the input="filename" option
MyList=["C:\\Users\\profa\\Documents\\Python Scripts\\TextMining\\Week2_3\\PretendCorpus\\T1.txt",
        "C:\\Users\\profa\\Documents\\Python Scripts\\TextMining\\Week2_3\\PretendCorpus\\T2.txt",
        "C:\\Users\\profa\\Documents\\Python Scripts\\TextMining\\Week2_3\\PretendCorpus\\T3.txt"]
## 
## Here, we have created a list of CONTENT
## We use a list of CONTENT when we use CountVectorizer with the input="content" option
## See both examples below and try both.
## Then read below for the next eaxmple....
MyContentList=["This is all the text that might be some document one", 
               "Another document might have other text in it, like this.",
               "Each document can be a review, or a website, or a novel, etc. "]
#
# Create a CountVectorizer for you to use. This is called "instantiation"
## This CountVecotizer will be used with MyList, the list of file locations. 
MyCV1 = CountVectorizer(input = "filename", stop_words="english")

## This CountVectorizer will be used with MyContentList. 
## Notice the difference between the two!
MyCV2 = CountVectorizer(input="content", stop_words="english")            

## Now - let's use fit.transform to transform the list of files                                 
MyMat = MyCV1.fit_transform(MyList)
print(type(MyMat))
## Get the column names
MyCols=MyCV1.get_feature_names()
print(MyCols)
## Create the dataframe...
MyDF = pd.DataFrame(MyMat.toarray(), columns=MyCols)
print(MyDF)


###
### Now - let's do this all over again using MyCV2 and the list of content....
MyMat2 = MyCV2.fit_transform(MyContentList)
print(type(MyMat2))
## Get the column names
MyCols2=MyCV2.get_feature_names()
print(MyCols2)
## Create the dataframe...
MyDF2 = pd.DataFrame(MyMat2.toarray(), columns=MyCols2)
print(MyDF2)

#############################################################################
#####################  NEXT and more challenging EXAMPLE ####################
#############################################################################

## Now - the above is fine - but not as useful because we either need to type 
## in the content - which is generally not possible OR
## We need to hard-code (type in) the file paths.

## Instead - we want this to be DYNAMIC

## So - create a corpus called PretendCorpus  or whatver you want to name it
## fill it up with a few (at least 4 or 5) .txt files that each have text
## in them. You can make this up or use reviews for the text - or whatver you like

## Now - let's use the operating system in Python to access the files rather
## than typing them in....

## Let's try it first - before we write a loop....
PathToMyCorpus="C:\\Users\\profa\\Documents\\Python Scripts\\TextMining\\Week2_3\\PretendCorpus"

MyFileNames = os.listdir(PathToMyCorpus)
print(MyFileNames)

## OK! Now we can access all the file names in any corpus. So if there are 10,000 files 
## in the corpus, we do not need to type them all into a list :)

## BUT ##

## We will now need to BUILD a list of all the complete file paths.
## To do this, I will start with an empty list

MyNewList=[]

## Next, I will create a for loop that will loop through all the files in my
## corpus (no matter how many there are) and I will put their paths into my new
## list....

for next_file_name in MyFileNames:
    ## BUILD the complete path by appending the path name to \\ to the filename
    CompleteFilePathName=PathToMyCorpus + "\\" + next_file_name 
    ## have a look: Looks good!
    print(CompleteFilePathName)
    ## Append each complete file path to your list
    MyNewList.append(CompleteFilePathName)
    

## See if the for loop built a list of all complete file paths....
print(MyNewList)

## YES!! ## - 
## As a challenge - try to do this for two corpuses

## Now we can use CountVecotirizer again
MyMat3 = MyCV1.fit_transform(MyNewList)
print(type(MyMat3))
## Get the column names
MyCols3=MyCV1.get_feature_names()
print(MyCols3)
## Create the dataframe...
MyDF3 = pd.DataFrame(MyMat3.toarray(), columns=MyCols3)
print(MyDF3)

## Good!
## Now we can do this for any number of files.
## Try this again, but add 4 more files to the corpus. 

## I will post an example using csv shortly - as well as labels, re and more...
