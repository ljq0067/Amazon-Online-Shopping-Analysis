import json
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

fashion_review = []
rate = []
reviews = []
df = pd.DataFrame()
stop_words = stopwords.words('english')

with open(
        'D:/Jieqian Liu/Data Science and Analystics/ANLY501 Data Science & Analytics/Individual Project Profolio/data gathering/Download_data/AMAZON_FASHION.json') as f:
    for line in tqdm(f):
        fashion_review = json.loads(line)
        # reviews.append(fashion_review['reviewText'])
        rate.append(fashion_review['overall'])
        reviews.append(fashion_review['reviewText'])
        if len(rate) == 575: break

#vectorizing the data to get word counts
#omitting words that appear less than 50 times or include non-English characters
regex1 = '[a-zA-Z]{6,100}'
MyCV_content = CountVectorizer(input='content', stop_words='english', token_pattern = regex1, min_df = 2 )

#creating a dataframe with the counts
My_DTM2 = MyCV_content.fit_transform(reviews)
ColNames = MyCV_content.get_feature_names()
My_DF_content = pd.DataFrame(My_DTM2.toarray(),columns=ColNames)

DFnoLabel = My_DF_content

#inserting the labels (countries) back into the data
My_DF_content.insert(loc=0, column='LABEL', value=rate)

DF = My_DF_content.loc[My_DF_content['LABEL'].isin(rate)]
#create training and test data
DF.to_csv('review_count1.csv', index=False, header=True, encoding='utf-8')


