import requests
import json

with open('topseller.json', 'r') as load_f:
    amazon_id = json.load(load_f)
asinList = amazon_id['sellerIdList']
asinStr = ','.join(asinList).replace(' [ ', ' ').replace(' ] ', ' ')
sellerId = asinStr[169603: 171077]
print(len(sellerId))

url = f'https://api.keepa.com/seller?key=7shsdsmam3tg745bicbo6eek3l0h9hf0a7i06a1otjpsb05d188kslln8h0jtsds&domain=1&seller={sellerId}'
req = requests.post(url)
with open(f'seller/seller114.json', 'wb') as f:
    f.write(req.content)

# Example = pd.read_csv('data.csv', parse_dates=['date'], index_col='date')
