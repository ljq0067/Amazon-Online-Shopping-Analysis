# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 13:57:11 2020

@author: profa
"""

## Example2.py
## CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

MyStops=['thi']
MyCV1 = CountVectorizer(input = "filename", stop_words=MyStops)

#MyDocs = ["This is document 1", "This is document 2", "Here is the Doc 3"]
## If you use the above, you need to change input to "content" and the variable MyList
## below to MyDocs. In other words, make sure each line of code makes sense. 

## Finally, my paths are NOT your paths - you will need to update these....

MyList=["C:\\Users\\profa\\Documents\\Python Scripts\\TextMining\\Week2_3\\PretendCorpus\\T1.txt",
        "C:\\Users\\profa\\Documents\\Python Scripts\\TextMining\\Week2_3\\PretendCorpus\\T2.txt",
        "C:\\Users\\profa\\Documents\\Python Scripts\\TextMining\\Week2_3\\PretendCorpus\\T3.txt"]


MyMat = MyCV1.fit_transform(MyList)
print(type(MyMat))

MyCols=MyCV1.get_feature_names()

print(MyCols)

MyDF = pd.DataFrame(MyMat.toarray(), columns=MyCols)
print(MyDF)

