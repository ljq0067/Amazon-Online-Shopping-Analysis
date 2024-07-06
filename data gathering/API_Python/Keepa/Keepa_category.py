import requests


url = f'https://api.keepa.com/category?key=7shsdsmam3tg745bicbo6eek3l0h9hf0a7i06a1otjpsb05d188kslln8h0jtsds&domain=1&category=0&parents=1'
req = requests.get(url)
with open(f'category.json', 'wb') as f:
    f.write(req.content)


