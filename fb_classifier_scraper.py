import time
import pandas as pd
import bs4
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup

first_name = []
last_name = []
state = []
grad_year = []
position = []
height = []
weight = []
forty_yard_dash = []
shuttle_run = []
three_cone = []
broad_jump = []
vertical_jump = []
profile_yn = []
def get_df(url):
    hdr = {'User-Agent': 'Mozilla/5.0'}
    req = Request(url, headers=hdr)
    page = urlopen(req)
    soup = BeautifulSoup(page, 'html.parser')
    rows = soup.find_all('tr')
    for row in rows:
        print(row)
        try:
            cols=row.find_all('td')
            cols=[x.text.strip() for x in cols]
            first_name.append(cols[0])
            last_name.append(cols[1])
            state.append(cols[2])
            grad_year.append(cols[3])
            position.append(cols[4])
            height.append(cols[5])
            weight.append(cols[6])
            forty_yard_dash.append(cols[7])
            shuttle_run.append(cols[8])
            three_cone.append(cols[9])
            broad_jump.append(cols[10])
            vertical_jump.append(cols[11])
            profile_yn.append(cols[12])
        except:
            pass
    df = pd.DataFrame(
    {'first_name': first_name,
     'last_name': last_name,
     'state': state,
     'grad_year': grad_year,
     'position': position,
     'height': height,
     'weight': weight,
     'forty_yard_dash': forty_yard_dash,
     'shuttle_run': shuttle_run,
     'three_cone': three_cone,
     'broad_jump': broad_jump,
     'vertical_jump': vertical_jump,
     'profile_yn': profile_yn,
    })
    df.to_csv('miami_combine_results.csv')

get_df('https://www.ncsasports.org/football/combine-results/2018-miami-rivals-combine-results')
