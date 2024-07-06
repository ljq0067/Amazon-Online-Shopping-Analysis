import json
from tqdm import tqdm_notebook as tqdm
import pandas as pd

with open('D:/Jieqian Liu/Data Science and Analystics/ANLY501 Data Science & Analytics/Individual Project Profolio/data gathering/API_Python/Keepa/product_after/product1.json', 'r', encoding='gb18030', errors='ignore') as f:
    data = json.load(f)

product = data['products']
amazon = []
title = []
rank = []
for i in range(len(product)):
    history = product[i]
    csv = history['csv']
    title.append(history['title'])
    rank.append(history['salesRanks'])
    a = csv[0]
    x = []
    for i in range(len(a)):
        if 0 < a[i] <= 10000:
            x.append(a[i])
    amazon.append(x)


sale_rank = []
cat = []
for n in rank:
    m = [key for key in n.keys()]
    cat.append(m)
    sales = []
    for key in n.keys():
        value = n[key]
        y = []
        for i in range(len(value)):
            if 0 < value[i] <= 10000:
                y.append(value[i])
        sales.append(y)
    sale_rank.append(sales)

data1 = pd.DataFrame(title)
data2 = pd.DataFrame(cat)
data3 = pd.DataFrame(sale_rank)
data1.to_csv("title.csv", index=False, header=True, encoding='utf-8')
data2.to_csv("category.csv", index=False, header=True, encoding='utf-8')
data3.to_csv("sales_rank.csv", index=False, header=True, encoding='utf-8')
