import time
import pandas as pd
from urllib.request import urlopen
from tqdm.notebook import tqdm
import requests
from bs4 import BeautifulSoup

import warnings
warnings.simplefilter('ignore', FutureWarning)

class Get_Jockey_Data(object):
    def __init__(self):
        self.init_url = 'https://db.netkeiba.com//?pid=jockey_leading'
        self.save_folder = 'Data/Jockey/'

    def main(self, start_year, end_year):
        year_list = list(range(start_year, end_year+1, 1))

        for year in tqdm(year_list):
            jockey_data = pd.DataFrame([])
            page = 1
            while True:
                time.sleep(0.5)
                url = f'https://db.netkeiba.com//?pid=jockey_leading&year={year}&page={page}'
                response = urlopen(url)
                df = pd.read_html(response.read(), header=1)[0]

                if df.empty:
                    break

                response = requests.get(url)
                html_content = response.content
                soup = BeautifulSoup(html_content, "html.parser")
                table = soup.find('table')
                jockey_ids = []
                for tr in table.find_all('tr')[2:]:
                    for td in tr.find_all('td', attrs={'class':'txt_l'}):
                        tmp_id = td.a['href']

                        if 'jockey' in tmp_id:
                            jockey_ids.append([item for item in tmp_id.split('/') if item != ''][-1])

                df['jockey_id'] = jockey_ids
                jockey_data = pd.concat([jockey_data, df], axis=0)
                page += 1

            jockey_data = jockey_data.reset_index(drop=True)
            jockey_data.to_pickle(self.save_folder + f'{year}.pkl')

    