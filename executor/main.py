#inference 
import sys
sys.path.append('/home/minkescanor/Desktop/WORKPLACE/EDABK/Human Img Classify/Human-IMG-Classification/results')

import pandas as pd
import requests
from bs4 import BeautifulSoup as bs

url = 'https://www.dior.com/en_vn'

html = requests.get(url=url)
html = str(html.text)

image_bs = bs(html, 'html5lib')

lst = image_bs.findAll('img')

print(len(lst))
