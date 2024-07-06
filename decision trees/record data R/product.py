import json
import pandas as pd

with open('D:/Jieqian Liu/Data Science and Analystics/ANLY501 Data Science & Analytics/Individual Project Profolio/data gathering/API_Python/Keepa/product_after/product10.json', 'r', encoding='gb18030', errors='ignore') as f:
    data = json.load(f)

product = data['products']
amazon = []
title = []
brand = []
type = []
asin = []
for i in range(len(product)):
    history = product[i]
    csv = history['csv']
    title.append(history['title'])
    brand.append(history['brand'])
    type.append(history['type'])
    asin.append(history['asin'])
    a = csv[0]
    x = []
    for i in range(len(a)):
        if 0 < a[i] <= 10000:
            x.append(a[i])
    amazon.append(x)


price = pd.DataFrame(amazon)
dict = {"asin": asin, "title": title, "brand": brand, "type": type}
data1 = pd.DataFrame(dict)
data2 = pd.concat([data1, price], axis=1)
data2.to_csv("product10.csv", index=False, header=True, encoding='utf-8')

