# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 13:26:55 2020

@author: profa
"""
## LiveExample.py

from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import pandas as pd
import re


#PretendData=["This is document 1", "This can be document 2", "Here is document 3"]
PretendCorpus="PretendCorpus"
MyList=["C:\\Users\\profa\\Documents\\Python Scripts\\TextMining\\Week2_3\\PretendCorpus\\T1.txt",
        "C:\\Users\\profa\\Documents\\Python Scripts\\TextMining\\Week2_3\\PretendCorpus\\T2.txt",
        "C:\\Users\\profa\\Documents\\Python Scripts\\TextMining\\Week2_3\\PretendCorpus\\T3.txt",
        "C:\\Users\\profa\\Documents\\Python Scripts\\TextMining\\Week2_3\\PretendCorpus\\T4.txt"]

#print(MyList)

from nltk.stem.porter import PorterStemmer

STEMMER=PorterStemmer()

print(STEMMER.stem("fishers"))

# Use NLTK's PorterStemmer in a function
def MY_STEMMER(str_input):
    words = re.sub(r"[^A-Za-z\-]", " ", str_input).lower().split()
    words = [STEMMER.stem(word) for word in words]
    return words

## Create a CountVectorizer object that you can use
MyCV1 = CountVectorizer(input="filename", 
                        #stop_words='english', 
                        tokenizer=MY_STEMMER,
                        lowercase=True)

## Call your MyCV1 on the data
DTM1 = MyCV1.fit_transform(MyList)
## get col names
ColNames=MyCV1.get_feature_names()
print(ColNames)

## convert DTM to DF

MyDF1 = pd.DataFrame(DTM1.toarray(), columns=ColNames)
print(MyDF1)


