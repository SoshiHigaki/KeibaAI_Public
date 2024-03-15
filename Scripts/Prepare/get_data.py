import requests
from bs4 import BeautifulSoup
import re
import time
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np


class get_data(object):
    
    def __init__(self):
        self.race_url_pattern = r'/race/\d+'
        self.init_race_url = 'https://db.netkeiba.com/'

    # def main(self, start_year, start_mon, end_year, end_mon):
    #     base_url = self.get_base_url(start_year, start_mon, end_year, end_mon)
    #     race_urls = self.get_race_url(base_url)


    
    def get_base_url(self, start_year, start_mon, end_year, end_mon):
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
    

    def get_each_race(self, url):
        response = requests.get(url)
        html = response.content
        soup = BeautifulSoup(html, 'html.parser')

        df = self.get_race_table(soup)
        info = self.get_info_table(soup)

        df = self.concate_data(df, info)

        df = self.clean(df)

        return df
    
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
                        id = re.findall(r'\d+', tmp)[0]
                        horse_ids.append(id)
                    elif '/jockey/' in tmp:
                        id = re.findall(r'\d+', tmp)[0]
                        jockey_ids.append(id)
                    elif '/trainer/' in tmp:
                        id = re.findall(r'\d+', tmp)[0]
                        trainer_ids.append(id)
                except:
                    pass

            data.append(row)

        # DataFrameを作成
        df = pd.DataFrame(data, columns=headers)
        df = df.loc[:, ['着順', '枠番', '馬番', '性齢', '斤量', 'タイム', '上り', '単勝', '人気', '馬体重']]

        df['horse_id'] = horse_ids
        df['jockey_id'] = jockey_ids
        df['trainer_id'] = trainer_ids

        return df
    
    
    def get_info_table(self, soup):
        race_info = soup.find('diary_snap').find('diary_snap_cut').find('span').text.strip()

        length_type = race_info.split('\xa0/\xa0')[0]
        weather = race_info.split('\xa0/\xa0')[1]
        condition = race_info.split('\xa0/\xa0')[2]

        cource_type = length_type[0]
        length = re.findall(r'\d+', length_type)[0]
        weather = weather.split(':')[1].strip()
        condition = condition.split(':')[1].strip()

        return [cource_type, length, weather, condition]
    
    def concate_data(self, df, info):
        info_df = pd.DataFrame([])
        info_df['cource_type'] = [info[0]] * len(df)
        info_df['length'] = [info[1]] * len(df)
        info_df['weather'] = [info[2]] * len(df)
        info_df['condtion'] = [info[3]] * len(df)

        df = pd.concat([df, info_df], axis=1)

        return df
    
    def clean(self, df):
        df['gender'] = df['性齢'].apply(self.get_gender) 
        df['age'] = df['性齢'].apply(self.get_age)
        df['weight'] = df['馬体重'].apply(self.get_weight)
        df['weight_diff'] = df['馬体重'].apply(self.get_weight_diff)
        df = df.drop(['性齢', '馬体重'], axis=1)

        df['time'] = df['タイム'].apply(self.get_second)
        df = df.drop(['タイム'], axis=1)

        return df


    def get_gender(self, x):
        try:
            gender = x[0]
        except:
            gender = np.nan
        return gender
    
    def get_age(self, x):
        try:
            age = x[1]
        except:
            age = np.nan
        return age

    def get_weight(self, x):
        try:
            match = re.match(r'(\d+)(\(\+\d+\))?', x)
            weight = match.group(1) 
        except:
            weight = np.nan
        return weight
    
    def get_weight_diff(self, x):
        try:
            match = re.match(r'(\d+)(\(\+\d+\))?', x)
            weight_diff = match.group(2).strip('()')
        except:
            weight_diff = 0
            
        return weight_diff
        
    def get_second(self, x):
        try:
            time_parts = x.split(':')
            minutes = int(time_parts[0])
            seconds = float(time_parts[1])

            total_seconds = minutes * 60 + seconds
        
        except:
            total_seconds = np.nan

        return total_seconds

                        
        