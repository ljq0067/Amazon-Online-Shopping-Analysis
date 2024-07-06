##TwitterMining_Token_WordCloud.py
## Gates
##
## You must have a Twitter Dev account and the 4 code to use this

## Twitter is not required in this class - but it is strongly recommended as it is common in industry


###Packages-----------------------
import pandas as pd
import tweepy
# conda install -c conda-forge tweepy
from tweepy import OAuthHandler
import json
from tweepy import Stream
from tweepy.streaming import StreamListener
import sys

from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer

from os import path
# from scipy.misc import imread
import matplotlib.pyplot as plt
##install wordcloud
## conda install -c conda-forge wordcloud
## May also have to run conda update --all on cmd
# import PIL
# import Pillow
# import wordcloud
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

###-----------------------------------------


## All 4 keys are in my TwitterCodesFile.txt and are comma sep
filename = "D://Jieqian Liu//Data Science and Analystics//ANLY501 Data Science & Analytics//Individual Project Profolio//data gathering//Twitter_API_Keys.txt"
with open(filename, "r") as FILE:
    keys = [i for line in FILE for i in line.split(',')]

# API Key:
consumer_key = keys[0]
# API Secret Key:
consumer_secret = keys[1]
# Access Token:
access_token = keys[2]
# Access Token Secret:
access_secret = keys[3]

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)


##-----------------------------------------------------------------
# Other Tweepy options - FYI
# for status in tweepy.Cursor(api.home_timeline).items(10):
# Process a single status
#   print(status.text)
#
# def Gather(tweet):
#   print(json.dumps(tweet))
# for friend in tweepy.Cursor(api.friends).items():
#   Gather(friend._json)
# --------------------------------------------------------------


class Listener(StreamListener):
    print("In Listener...")
    tweet_number = 0

    # __init__ runs as soon as an instance of the class is created
    def __init__(self, max_tweets, hfilename, rawfile, TweetsList, LabelsList, nohashname):
        self.max_tweets = max_tweets
        print(self.max_tweets)
        # on_data() is a function of StreamListener as is on_error and on_status

    def on_data(self, data):
        self.tweet_number += 1
        print("In on_data", self.tweet_number)
        try:
            print("In on_data in try")
            with open(hfilename, 'a') as f:
                with open(rawfile, 'a') as g:
                    tweet = json.loads(data)
                    ## RE: https://realpython.com/python-json/
                    tweet_text = tweet["text"]
                    print(tweet_text, "\n")
                    TweetsList.append(tweet_text)
                    LabelsList.append(nohashname)
                    print(TweetsList)
                    f.write(tweet_text)  # the text from the tweet
                    json.dump(tweet, g)  # write the raw tweet
        except BaseException:
            print("NOPE")
            pass
        if self.tweet_number >= self.max_tweets:
            # sys.exit('Limit of '+str(self.max_tweets)+' tweets reached.')
            print("Got ", str(self.max_tweets), "tweets.")
            return False

    # method for on_error()
    def on_error(self, status):
        print("ERROR")  # machi
        print(status)  # 401 your keys are not working
        if (status == 420):
            print("Error ", status, "rate limited")
            return False


# ----------------end of class Listener

hashname = input("Enter the hash name, such as #womensrights: ")
numtweets = eval(input("How many tweets do you want to get?: "))
if (hashname[0] == "#"):
    nohashname = hashname[1:]  # remove the hash
else:
    nohashname = hashname
    hashname = "#" + hashname

# Create a file for any hash name
hfilename = "file_" + nohashname + ".txt"
## FOr example, file_football.txt  if you used #football
rawfile = "file_rawtweets_" + nohashname + ".txt"
## For example, file_rawtweets_football.txt
## Notice that the raw file is in json
## The hfilename is just text
TweetsList = []
LabelsList = []

################ Get the tweets..................................
twitter_stream = Stream(auth, Listener(numtweets,
                                       hfilename, rawfile,
                                       TweetsList, LabelsList,
                                       nohashname))
## https://developer.twitter.com/en/docs/twitter-api/tweets/filtered-stream/introduction
## https://developer.twitter.com/en/docs/twitter-api/enterprise/powertrack-api/overview
# twitter_stream.filter(track=['#womensrights'])
twitter_stream.filter(track=[hashname])
##..................................................................

## Save each Tweet in a list
## This will create a LIST OF CONTENT
## that you can use with CountVectorizer, etc.

print(TweetsList)
print(LabelsList)

MyCV_T = CountVectorizer(input='content',
                         stop_words='english',
                         # max_features=100
                         )

My_DTM_T = MyCV_T.fit_transform(TweetsList)

## BUT - we are not done! Right now, we havea DTM.
## We actually want a dataframe.
## Let's convert our DTM to a DF

## TWO Steps:
## First - use your CountVectorizer to get all the column names
ColNames = MyCV_T.get_feature_names()
print("The vocab is: ", ColNames, "\n\n")

## NEXT - Use pandas to create data frames
My_DF_T = pd.DataFrame(My_DTM_T.toarray(), columns=ColNames)

## Let's look!
print(My_DF_T)
print(LabelsList)

## Remove all columns that contain any numbers/digits
# for i in My_DF_T.columns:
#    print(i)

droplist = [i for i in My_DF_T.columns if re.search(r'\d', i)]
print(droplist)
My_DF_T.drop(droplist, axis=1, inplace=True)

print(My_DF_T)

### Let's create a csv file that we can write this DF
## into AND that we can reuse each time we run this
## for a different #.
## This is one way to create LABELED text data

## Check if file exists and if not create it

OutFile_TweetDF = "Labeled_Tweets.csv"

## Write what we have so far into the file

## BUT first - let's add the LABEL to our dataframe in the front
## before we write the DF to the file.
## This way, each time we run this code with a different #
## it will create labeled data and will add it to this
## same csv file.

## Because the number of tweets can vary - the easiest way to do
## this is to create a list of labels as you collect tweets.
## this will be done ABOVE in the object.
## This is called LabelsList

## Add the list to the dataframe to create labels.
My_DF_T.insert(loc=0, column='LABEL', value=LabelsList)
print(My_DF_T)

# import os
# if file does not exist write header
if not os.path.isfile(OutFile_TweetDF):
    print("does not exist...creating file now...")
    My_DF_T.to_csv(OutFile_TweetDF)

else:  # else it exists so append without writing the header
    print("file exists - appending to it....")
    My_DF_T.to_csv(OutFile_TweetDF, mode='a', header=False)

## Now - run the code again with a different #

## Each time you run the code with a different #whatever

## The csv file will append the results.