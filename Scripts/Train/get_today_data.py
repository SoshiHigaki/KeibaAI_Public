import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
from datetime import datetime
from urllib.parse import urlparse, parse_qs

from Scripts.Prepare import get_horse_data

class TodaysData(object):
    def __init__(self):
        self.get_horse_data = get_horse_data.get_horse_data()
        self.save_path = 'Data/today.pkl'
        self.horse_names_path = 'Data/horse_names.txt'
        self.horse_numbers_path = 'Data/horse_numbers.txt'
    
    def main(self, url):
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        race_id = query_params.get('race_id')[0]

        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        df = self.get_raw_table(soup)
        info = self.get_race_info(soup)
        date = self.get_date(soup)

        df = self.clean(df)
        df = self.concat(df, info, date, race_id)
        df = self.select_columns(df)

        df = self.set_type(df)

        horse_ids = list(df['horse_id'])
        self.get_horse_data.main(horse_ids)

        df.to_pickle(self.save_path)


    def get_raw_table(self, soup):
        
        table = soup.find('div', attrs={'class':'RaceTableArea'})
                        
        headers = []
        data = []

        for th in table.find_all('th'):
            headers.append(th.text.strip())

        horse_ids = []
        jockey_ids = []
        trainer_ids = []
        for tr in table.find_all('tr')[2:]:
            horse_url = tr.find_all('a')[0].get('href')
            horse_id = [x for x in horse_url.split('/') if x != ''][-1]
            horse_ids.append(horse_id)

            jockey_url = tr.find_all('a')[1].get('href')
            jockey_id = [x for x in jockey_url.split('/') if x != ''][-1]
            jockey_ids.append(jockey_id)

            trainer_url = tr.find_all('a')[2].get('href')
            trainer_id = [x for x in trainer_url.split('/') if x != ''][-1]
            trainer_ids.append(trainer_id)

            row = []
            for td in tr.find_all('td'):
                row.append(td.text.strip())
            data.append(row)

        headers.remove('お気に入り馬')
        df = pd.DataFrame(data, columns=headers)
        df['horse_id'] = horse_ids
        df['jockey_id'] = jockey_ids
        df['trainer_id'] = trainer_ids


        np.savetxt(self.horse_names_path, np.array(df['馬名'], dtype=str), fmt='%s')
        np.savetxt(self.horse_numbers_path, np.array(df['馬番'], dtype=int), fmt='%d')

        df = df.loc[:, ['horse_id', 'jockey_id', 'trainer_id', '馬番', '性齢', '斤量', '馬体重(増減)']]
        df.columns = ['horse_id', 'jockey_id', 'trainer_id', 'horse_number', 'sex_and_age', 'weight_carried', 'horse_weight']

        return df


    def get_race_info(self, soup):
        info = soup.find('div', attrs={'class':'RaceData01'}).text.strip()
        
        try:
            type_length = info.split('/')[1].split('(')[0].replace(" ", "")
            type = type_length[0]
            length = re.findall(r'\d+', type_length)[0]
        except:
            type = np.nan
            length = np.nan
            
        try:
            weather = info.split('/')[2].split(':')[1][0]
        except:
            weather = np.nan

        try:
            condition = info.split('/')[3].split(':')[1]
        except:
            condition = np.nan

        return [type, length, weather, condition]
    
    def clean(self, df):
        df['sex'] = df['sex_and_age'].apply(self.get_sex) 
        df['age'] = df['sex_and_age'].apply(self.get_age)
        df['weight'] = df['horse_weight'].apply(self.get_weight)
        df['weight_difference'] = df['horse_weight'].apply(self.get_weight_diff)
        df = df.drop(['sex_and_age', 'horse_weight'], axis=1)

        return df
    
    def concat(self, df, info, date, race_id):
        info_df = pd.DataFrame([])
        info_df['type'] = [info[0]] * len(df)
        info_df['length'] = [info[1]] * len(df)
        info_df['weather'] = [info[2]] * len(df)
        info_df['condition'] = [info[3]] * len(df)

        info_df['race_id'] = [race_id] * len(df)
        info_df['date'] = [date] * len(df)

        info_df['velocity'] = [0] * len(df)

        df = pd.concat([df, info_df], axis=1)

        return df

    def get_sex(self, x):
        try:
            sex = x[0]
        except:
            sex = np.nan
        return sex
    
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
            x = x.split('(')[1]
            x = x.split(')')[0]
            weight_diff = int(x)
        except:
            weight_diff = np.nan
        return weight_diff
    
    def select_columns(self, df):
        df = df.loc[:, ['race_id', 'horse_id', 'jockey_id', 'trainer_id', 'date', 
                        'horse_number', 'weight_carried', 'type', 'length', 'weather',
                        'condition', 'sex', 'age', 'weight', 'weight_difference', 'velocity']]
        return df
    
    def get_date(self, soup):
        day = soup.find('dd', attrs={'class':'Active'}).find('a').get('title').split('(')[0]
        date_string = day.replace('月', '-').replace('日', '')
        date_string = date_string.strip()

        # 日付文字列をdatetime型に変換
        date_format = '%m-%d'
        race_date = datetime.strptime(date_string, date_format)
        target_month = race_date.month
        target_day = race_date.day

        today = datetime.now().date()
        target_date = datetime(today.year, target_month, target_day).date()
        if target_date < today:
            target_date = datetime(today.year + 1, target_month, target_day).date()

        return target_date
    
    def set_type(self, df):

        df['race_id'] = df['race_id'].astype(str, errors='ignore')
        df['horse_id'] = df['horse_id'].astype(str, errors='ignore')
        df['jockey_id'] = df['jockey_id'].astype(str, errors='ignore')
        df['trainer_id'] = df['trainer_id'].astype(str, errors='ignore')

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