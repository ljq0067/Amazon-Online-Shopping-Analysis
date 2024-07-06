import requests


def get_data():
    url = f'https://api.keepa.com/topseller?key=7shsdsmam3tg745bicbo6eek3l0h9hf0a7i06a1otjpsb05d188kslln8h0jtsds&domain=1'
    req = requests.get(url)
    with open(f'topseller.json', 'wb') as f:
        f.write(req.content)


get_data()
