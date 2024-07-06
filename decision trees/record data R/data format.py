import pandas as pd

data = pd.read_csv("product.csv", encoding='utf-8')
y = {}
price = pd.DataFrame()
x = {}
for i in range(91):
    for j in range(20):
        y[j] = data.iloc[i, j+5] / data.iloc[i, j+4]
    x = data.iloc[i, 24]/data.iloc[i, 4]
    if x > 1: y[20] = "up"
    if x < 1: y[20] = "down"
    if x == 1: y[20] = "same"

    price = price.append(pd.DataFrame(y, index=[i]))

df = pd.concat([data.iloc[:, 0:4], price], axis=1)
df = df.drop('title', 1)
df.to_csv('price_label.csv', index=False, header=True, encoding='utf-8')