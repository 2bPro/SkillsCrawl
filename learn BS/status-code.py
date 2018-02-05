from bs4 import BeautifulSoup

import requests

result = requests.get("https://jessesw.com/Data-Science-Skills/")

print result

c = result.content

soup = BeautifulSoup(c, "lxml")

samples = soup.find_all("h1")

print samples[0]
