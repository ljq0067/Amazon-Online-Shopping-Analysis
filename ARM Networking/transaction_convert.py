import spacy
from spacy.language import Language
import ray
# import psutil
import modin.pandas as pd
from tqdm import tqdm
# from modin.config import ProgressBar

ray.init()

df = pd.read_csv(
    'D:/Jieqian Liu/Data Science and Analystics/ANLY501 Data Science & Analytics/Individual Project Profolio/data/Reviews.csv')
df = df[df['Text'].str.len() > 0]
df.head()

pipeline = spacy.load('en_core_web_sm')

texts = df['Text']
doc1 = pipeline(texts[2])
for i, token in enumerate(doc1):
    print({"text": token.text,
           "lemma": token.lemma_,
           "POS": token.pos_,
           "tag": token.tag_,
           "dep": token.dep_,
           "shape": token.shape_,
           "is_alpha": token.is_alpha,
           "is_stop": token.is_stop})


@Language.component('remove_stop')
def remove_stop(doc):
    return [token.lemma_.lower().strip().replace("'", '') for token in doc if
            not token.is_stop and 1 < len(token.lemma_) < 25]


pipeline.add_pipe('remove_stop')

pipeline.analyze_pipes(pretty=True)

print('origin text: ', texts[2])
print(pipeline(texts[2]))


def f(t):
    tokens = pipeline(t)
    return tokens


texts = texts.apply(f)

f = open('transaction.txt', 'w')
for tokens in tqdm(texts):
    if len(tokens) < 2:
        continue
    f.write(','.join(tokens))
    f.write('\n')
f.close()
