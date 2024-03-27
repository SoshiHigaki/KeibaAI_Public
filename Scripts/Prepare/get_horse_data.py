import requests
from bs4 import BeautifulSoup
import re
import time
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
from datetime import datetime
import os

import warnings
warnings.simplefilter('ignore', FutureWarning)

class get_horse_data(object):
    def __init__(self):
        self.init_url = 'https://db.netkeiba.com/horse/'
        self.save_folder = 'Data/Horse/'
        os.makedirs(self.save_folder) if not os.path.exists(self.save_folder) else None

    def main(self, horse_ids):
        for id in tqdm(horse_ids):
            time.sleep(1)
            try:
                url = self.get_url(id)
                df = self.get_each_data(url)
                df = self.clean_horse_data(df)
                df = self.get_columns(df)
                df = self.set_type(df)

                path = f'{self.save_folder}{id}.pkl'
                df.to_pickle(path)
            except:
                print(f'エラー : {url}')
        
    

    def get_url(self, horse_id):
        url = self.init_url + str(horse_id)
        return url
    
    
    def get_each_data(self, url):
        response = requests.get(url)
        html_content = response.content
        soup = BeautifulSoup(html_content, 'html.parser')

        sex = soup.find('p', attrs={'class':'txt_01'}).text.replace('\u3000', ' ').split(' ')[1]
        birthday = soup.find('div', attrs={'class':'db_prof_area_02'}).find('td').text
        birthday = datetime.strptime(birthday, "%Y年%m月%d日")
        
        df = self.get_race_table(soup)
        df['sex'] = [sex] * len(df)
        df['birthday'] = [birthday] * len(df)
        
        return df
    
    def get_race_table(self, soup):
        table = soup.find('table', class_='db_h_race_results nk_tb_common')
        rows = table.find_all('tr')

        header_row = rows[0]
        headers = [header.text.strip() for header in header_row.find_all('th')]

        data = []
        for row in rows[1:]:
            values = [value.text.strip() for value in row.find_all('td')]
            data.append(values)

        df = pd.DataFrame(data, columns=headers)
        df = df.loc[:, ['日付', '天気', '馬番', '斤量', '距離', '馬場', 'タイム', '馬体重']]
        df.columns = ['date', 'weather', 'horse_number', 'weight_carried', 'length_and_type', 'condition', 'time', 'horse_weight']

        return df

    
    def clean_horse_data(self, df):
        
        df['date'] = df['date'].apply(self.get_date)
        df['type'] = df['length_and_type'].apply(self.get_type)
        df['length'] = df['length_and_type'].apply(self.get_length)
        df['time'] = df['time'].apply(self.get_second)
        df['weight'] = df['horse_weight'].apply(self.get_weight)
        df['weight_difference'] = df['horse_weight'].apply(self.get_weight_diff)

        df['velocity'] = df.apply(self.get_velocity, axis=1)
        df['age'] = df.apply(self.get_age, axis=1)
        
        return df
    
    def get_columns(self, df):
        df = df.loc[:, ['date', 'horse_number', 'weight_carried', 'type', 'length', 
                        'weather', 'condition', 'sex', 'age', 'weight', 'weight_difference', 'velocity']]
        
        return df
        
    def get_date(self, x):
        date = datetime.strptime(x, "%Y/%m/%d")
        return date

    def get_type(self, x):
        return x[0]

    def get_length(self, x):
        return re.findall(r'\d+', x)[0]

    def get_second(self, x):
        try:
            time_parts = x.split(':')
            minutes = int(time_parts[0])
            seconds = float(time_parts[1])

            total_seconds = minutes * 60 + seconds
            
        except:
            total_seconds = np.nan

        return total_seconds
    
    def get_weight(self, x):
        try:
            match = re.match(r'(\d+)(\(\+\d+\))?', x)
            weight = match.group(1) 
        except:
            weight = np.nan
        return weight
    
    def get_weight_diff(self, x):
        try:
            x = x.split('(')[1]
            x = x.split(')')[0]
            weight_diff = int(x)
        except:
            weight_diff = np.nan
        return weight_diff
    
    def get_velocity(self, row):
        time = float(row['time'])
        length = int(row['length'])
        velocity = length / time
        return velocity   
    
    def get_age(self, row):
        race_date = row['date']
        birthday = row['birthday']

        age =  race_date.year - birthday.year - ((race_date.month, race_date.day) < (birthday.month, birthday.day))

        return age
    
    def set_type(self, df):
        df['date'] = pd.to_datetime(df['date'], errors='ignore')
        df['horse_number'] = df['horse_number'].astype(int, errors='ignore')
        df['weight_carried'] = df['weight_carried'].astype(int, errors='ignore')
        df['type'] = df['type'].astype(str, errors='ignore')
        df['length'] = df['length'].astype(int, errors='ignore')
        df['weather'] = df['weather'].astype(str, errors='ignore')
        df['condition'] = df['condition'].astype(str, errors='ignore')
        df['sex'] = df['sex'].astype(str, errors='ignore')
        df['age'] = df['age'].astype(int, errors='ignore')
        df['weight'] = df['weight'].astype(int, errors='ignore')
        df['weight_difference'] = df['weight_difference'].astype(int, errors='ignore')
        df['velocity'] = df['velocity'].astype(float, errors='ignore')

        return df
