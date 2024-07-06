import json
from tqdm import tqdm_notebook as tqdm
import pandas as pd
import os
import glob
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import string

# fashion_review = []
# rating = []
# review = []
# with open(
#         'D:/Jieqian Liu/Data Science and Analystics/ANLY501 Data Science & Analytics/Individual Project Profolio/data gathering/Download_data/AMAZON_FASHION.json') as f:
#     for line in tqdm(f):
#         fashion_review = json.loads(line)
#         rating.append(fashion_review['overall'])
#         review.append(fashion_review['reviewText'])
#         if len(review) == 601:
#             break

os.chdir(r'.')
df = pd.DataFrame()
for filename in glob.glob('*.txt'):
    with open(filename, 'r') as file:
        txt = file.read()
        file.close()
        txt = re.sub('[?!;:,.\"\\-\\n]+', ' ', txt)
        txt = re.sub(' +', ' ', txt, flags=re.IGNORECASE)
        txt = re.sub(r'\\\"', ' ', txt, flags=re.IGNORECASE)
        txt = txt.replace(',', '')
        txt = txt.replace('0', '')
        txt = ' '.join(txt.split())
        df = df.append({'page_title': filename.split('.')[0], 'page_content': txt}, ignore_index=True)

for i in range(len(txt)):
    if txt[i] == 0:
        txt.rstrip('0')

rating = re.findall(r'\d+', txt)
review = txt.rstrip(string.digits)
review = ''.join(review)

review1 = []
review2 = []
review3 = []
review4 = []
review5 = []
for i in range(600):
    if rating[i] == 1.0:
        review1.append(review[i])
    if rating[i] == 2.0:
        review2.append(review[i])
    if rating[i] == 3.0:
        review3.append(review[i])
    if rating[i] == 4.0:
        review4.append(review[i])
    if rating[i] == 5.0:
        review5.append(review[i])

review1 = ''.join(review1)
review2 = ''.join(review2)
review3 = ''.join(review3)
review4 = ''.join(review4)
review5 = ''.join(review5)
CV = CountVectorizer()
wordmatrix = CV.fit_transform(review)
wordmatrix = pd.DataFrame(wordmatrix.toarray(),columns=CV.get_feature_names())
pd.set_option('display.max_columns',10)
print(wordmatrix.shape)
wordmatrix.head()
model = KMeans(n_clusters=3)
label = model.labels_
y_pred = model.fit_predict(wordmatrix)
y_pred

w = WordCloud(width=600, height=600, background_color="white").generate(review1)
w.to_file(r"review1.png")

w = WordCloud(width=600, height=600, background_color="white").generate(review2)
w.to_file(r"review2.png")

w = WordCloud(width=600, height=600, background_color="white").generate(review3)
w.to_file(r"review3.png")

w = WordCloud(width=600, height=600, background_color="white").generate(review4)
w.to_file(r"review4.png")

w = WordCloud(width=600, height=600, background_color="white").generate(review5)
w.to_file(r"review5.png")
