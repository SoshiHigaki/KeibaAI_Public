import requests
from bs4 import BeautifulSoup
import re
import time
from tqdm.notebook import tqdm
import pandas as pd


import requests
from bs4 import BeautifulSoup
import re
import time
from tqdm.notebook import tqdm

class get_data(object):
    
    def __init__(self):
        self.race_url_pattern = r'/race/\d+'
        self.init_race_url = 'https://db.netkeiba.com/'
    
    def base_url(self, start_year, start_mon, end_year, end_mon):
        url = f'https://db.netkeiba.com/?pid=race_list&word=&track%5B%5D=1&track%5B%5D=2&start_year={start_year}&start_mon={start_mon}&end_year={end_year}&end_mon={end_mon}&jyo%5B%5D=01&jyo%5B%5D=02&jyo%5B%5D=03&jyo%5B%5D=04&jyo%5B%5D=05&jyo%5B%5D=06&jyo%5B%5D=07&jyo%5B%5D=08&jyo%5B%5D=09&jyo%5B%5D=10&kyori_min=&kyori_max=&sort=date&list=100'
        return url
    
    def get_all_pages(self, base_url):
        response = requests.get(base_url)
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')

        text = str(soup.find('div', attrs={'class':'pager'}))
        pattern = re.compile(r'\d+')
        numbers = pattern.findall(text)

        if int(numbers[0]) % int(numbers[2]) == 0:
            all_pages = int(numbers[0]) // int(numbers[2])
        else:
            all_pages = int(numbers[0]) // int(numbers[2]) + 1

        return all_pages
        

    def get_race_url(self, base_url):
        all_pages = self.get_all_pages(base_url)

        race_urls = []
        for i in tqdm(range(1, all_pages+1, 1)):
            time.sleep(1)

            page_url = base_url + f'&page={i}'

            response = requests.get(page_url)
            html = response.text
            soup = BeautifulSoup(html, 'html.parser')

            pattern = re.compile(self.race_url_pattern)
            links = soup.find_all('a', href=pattern)

            norm_links = []
            for l in links:
                # href属性の値を取得
                href_value = l.get('href')
                # /race/から始まる部分を抽出
                race_url = href_value[href_value.find('/race/'):]

                tmp = self.init_race_url + race_url
                norm_links.append(tmp)

            race_urls.extend(norm_links)


        return race_urls
    
    def get_race_table(self, soup):
        table = soup.find('table', class_='race_table_01')
        # テーブルのヘッダーを取得
        headers = []
        for th in table.find_all('th'):
            headers.append(th.text.strip())

        # レース結果のデータを取得
        data = []
        horse_ids = []
        jockey_ids = []
        trainer_ids = []

        for tr in table.find_all('tr')[1:]:
            row = []
            for td in tr.find_all('td'):
                row.append(td.text.strip())
                
                try:
                    tmp = td.a['href']
                    if '/horse/' in tmp:
                        horse_ids.append(tmp)
                    elif '/jockey/' in tmp:
                        jockey_ids.append(tmp)
                    elif '/trainer/' in tmp:
                        trainer_ids.append(tmp)
                except:
                    pass

            data.append(row)

        # DataFrameを作成
        df = pd.DataFrame(data, columns=headers)
        df['horse_id'] = horse_ids
        df['jockey_id'] = jockey_ids
        df['trainer_id'] = trainer_ids

        return df
        
        