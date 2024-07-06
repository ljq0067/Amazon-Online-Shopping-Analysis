import json
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
import re
from nltk.corpus import stopwords

fashion_review = []
reviews = []
df = pd.DataFrame()
stop_words = stopwords.words('english')

with open(
        'D:/Jieqian Liu/Data Science and Analystics/ANLY501 Data Science & Analytics/Individual Project Profolio/data gathering/Download_data/AMAZON_FASHION.json') as f:
    for line in tqdm(f):
        fashion_review = json.loads(line)
        # reviews.append(fashion_review['reviewText'])
        reviews = fashion_review['reviewText']
        review = ''.join(reviews)
        review = review.lower()
        review = re.sub(r'[,.;@#?!&$\-\']+', ' ', review, flags=re.IGNORECASE)
        review = re.sub(' +', ' ', review, flags=re.IGNORECASE)
        review = re.sub(r'\"', ' ', review, flags=re.IGNORECASE)
        review = re.sub(r'[^a-zA-Z]', " ", review, flags=re.VERBOSE)
        review = review.replace(',', '')
        word = review.split()
        s1 = set(word)
        list1 = list(s1)
        list1 = [w for w in list1 if w not in stop_words]
        df.loc[len(reviews)-1, 'text'] = reviews
        for x in range(len(list1)):
            df.loc[len(reviews)-1, list1[x]] = 0
            for y in range(len(word)):
                if list1[x] == word[y]:
                    df.loc[len(reviews)-1, list1[x]] += 1
        if len(reviews) == 100: break

df = df.fillna(0)
series1 = df.sum()
x = series1[series1.values == 1].index
x = x.tolist()
df.drop(x, axis=1, inplace=True)
series2 = df.sum(axis=1)
x1 = series2[series2.values <= 1].index
x1 = x1.tolist()
df.drop(x1, inplace=True)

df.to_csv('review_count.csv', index=False, header=True, encoding='utf-8')


