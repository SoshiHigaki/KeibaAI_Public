import requests
from bs4 import BeautifulSoup
import re
import time
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
from datetime import datetime
import random


class Get_Data(object):
    
    def __init__(self):
        self.race_url_pattern = r'/race/\d+' # urlにヒットするパターン
        self.init_race_url = 'https://db.netkeiba.com/' # ネット競馬からデータを集めるので、ネット競馬の先頭のurl
        self.test_ratio = 0.2 # テストデータの割合は2割

    def main(self, start_year, start_mon, end_year, end_mon):
        base_url = self.get_base_url(start_year, start_mon, end_year, end_mon) # ページを指定する前のベースとなるurl
        race_urls = self.get_race_url(base_url) # ページを進めながらレースのurlを取得する

        test_size = int(len(race_urls) * self.test_ratio)
        random.shuffle(race_urls)
        train_race_urls = race_urls[test_size:] # train用のurl
        test_race_urls = race_urls[:test_size] # test用のurl

        whole_return_df = pd.DataFrame([])
        train_whole_df = pd.DataFrame([])
        for url in tqdm(train_race_urls, desc=f'{start_year}/{str(start_mon).zfill(2)}～{end_year}/{str(end_mon).zfill(2)} : Train'):
            time.sleep(0.5)
            try:
                df, return_df = self.get_each_race(url) # レースのデータと払い戻しのデータを取得
                train_whole_df = pd.concat([train_whole_df, df], axis=0)
                whole_return_df = pd.concat([whole_return_df, return_df], axis=0)
            except Exception as e:
                print(f'エラー : {url}')

        test_whole_df = pd.DataFrame([])
        for url in tqdm(test_race_urls, desc=f'{start_year}/{str(start_mon).zfill(2)}～{end_year}/{str(end_mon).zfill(2)} : Test'):
            time.sleep(0.5)
            try:
                df, return_df = self.get_each_race(url)
                test_whole_df = pd.concat([test_whole_df, df], axis=0)
                whole_return_df = pd.concat([whole_return_df, return_df], axis=0)
            except Exception as e:
                print(f'エラー : {url}')

        train_whole_df = self.clean(train_whole_df)
        train_whole_df = self.get_columns(train_whole_df)
        train_whole_df = self.set_type(train_whole_df)
        train_whole_df = train_whole_df.reset_index(drop=True)

        test_whole_df = self.clean(test_whole_df)
        test_whole_df = self.get_columns(test_whole_df)
        test_whole_df = self.set_type(test_whole_df)
        test_whole_df = test_whole_df.reset_index(drop=True)
        
        return train_whole_df, test_whole_df, whole_return_df


    # 期間を指定してurlを作成する関数
    def get_base_url(self, start_year, start_mon, end_year, end_mon):
        url = f'https://db.netkeiba.com/?pid=race_list&word=&track%5B%5D=1&track%5B%5D=2&start_year={start_year}&start_mon={start_mon}&end_year={end_year}&end_mon={end_mon}&kyori_min=&kyori_max=&sort=date&list=100'
        return url

    # データベースのページ数を取得
    def get_all_pages(self, base_url):
        response = requests.get(base_url)
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')

        text = str(soup.find('div', attrs={'class':'pager'}))
        pattern = re.compile(r'\d+')
        numbers = pattern.findall(text)
        
        number = int(''.join(numbers[:-2]))
        per = int(numbers[-1])

        if number % per == 0:
            all_pages = number // per
        else:
            all_pages = number // per + 1

        return all_pages
        
    # ページを進めながらレースのurlを取得する
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
    
    # レースのurlからレースの情報を取得する
    def get_each_race(self, url):
        race_id =  url.split('/')[-2]

        response = requests.get(url)
        html = response.content
        soup = BeautifulSoup(html, 'html.parser')

        date = soup.find('p', attrs={'class':'smalltxt'}).text.split()[0]
        date = datetime.strptime(date, '%Y年%m月%d日')

        df = self.get_race_table(soup)
        info = self.get_info_table(soup)

        df = self.concate_data(df, info, race_id, date)

        return_df = self.get_money_return(race_id, soup)

        return df, return_df

    # レースの情報をテーブルデータで取得する
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
                        id = tmp.split('/')[-2]
                        horse_ids.append(id)
                    elif '/jockey/' in tmp:
                        id = tmp.split('/')[-2]
                        jockey_ids.append(id)
                    elif '/trainer/' in tmp:
                        id = tmp.split('/')[-2]
                        trainer_ids.append(id)
                except:
                    pass

            data.append(row)

        # DataFrameを作成
        df = pd.DataFrame(data, columns=headers)
        df = df.loc[:, ['着順', '枠番', '馬番', '性齢', '斤量', 'タイム', '上り', '単勝', '人気', '馬体重']]
        df.columns = ['order', 'post_position', 'horse_number', 'sex_and_age', 'weight_carried',
                      'time', 'late_pace', 'win_odds', 'popularity', 'horse_weight']

        df['horse_id'] = horse_ids
        df['jockey_id'] = jockey_ids
        df['trainer_id'] = trainer_ids

        return df

    # レースが行われた時の環境に関する情報を取得する
    def get_info_table(self, soup):
        race_info = soup.find('diary_snap').find('diary_snap_cut').find('span').text.strip()

        length_type = race_info.split('\xa0/\xa0')[0]
        weather = race_info.split('\xa0/\xa0')[1]
        condition = race_info.split('\xa0/\xa0')[2]

        type = length_type[0]
        length = re.findall(r'\d+', length_type)[0]
        weather = weather.split(':')[1].strip()
        condition = condition.split(':')[1].strip()[0]

        return [type, length, weather, condition]

    # 各種情報をひとつのテーブルにまとめる
    def concate_data(self, df, info, race_id, date):
        info_df = pd.DataFrame([])
        info_df['type'] = [info[0]] * len(df)
        info_df['length'] = [info[1]] * len(df)
        info_df['weather'] = [info[2]] * len(df)
        info_df['condition'] = [info[3]] * len(df)

        info_df['race_id'] = [race_id] * len(df)
        info_df['date'] = [date] * len(df)

        df = pd.concat([df, info_df], axis=1)

        return df

    # データの形式を指定する
    def clean(self, df):
        df['sex'] = df['sex_and_age'].apply(self.get_sex) 
        df['age'] = df['sex_and_age'].apply(self.get_age)
        df['weight'] = df['horse_weight'].apply(self.get_weight)
        df['weight_difference'] = df['horse_weight'].apply(self.get_weight_diff)
        df = df.drop(['sex_and_age', 'horse_weight'], axis=1)

        df['time'] = df['time'].apply(self.get_second)
        df['velocity'] = df.apply(self.get_velocity, axis=1)

        return df

    # データとして利用するカラムを指定する
    def get_columns(self, df):
        df = df.loc[:, ['race_id', 'horse_id', 'jockey_id', 'trainer_id', 
                        'date', 'horse_number', 'weight_carried', 'type', 'length', 
                        'weather', 'condition', 'sex', 'age', 'weight', 'weight_difference', 'velocity']]
        
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
        
    def get_second(self, x):
        try:
            time_parts = x.split(':')
            minutes = int(time_parts[0])
            seconds = float(time_parts[1])

            total_seconds = minutes * 60 + seconds
        
        except:
            total_seconds = np.nan

        return total_seconds

    def get_velocity(self, row):
        time = float(row['time'])
        length = int(row['length'])
        velocity = length / time
        return velocity                
        
    def set_type(self, df):

        df['race_id'] = df['race_id'].astype(str, errors='ignore')
        df['horse_id'] = df['horse_id'].astype(str, errors='ignore')
        df['jockey_id'] = df['jockey_id'].astype(str, errors='ignore')
        df['trainer_id'] = df['trainer_id'].astype(str, errors='ignore')

        df['date'] = pd.to_datetime(df['date'], errors='ignore')
        df['horse_number'] = df['horse_number'].astype(int, errors='ignore')
        df['weight_carried'] = df['weight_carried'].astype(float, errors='ignore')
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

    # 払い戻しの情報を取得する
    def get_money_return(self, race_id, soup):
        table = soup.find('table', class_='pay_table_01')
        tmp = pd.DataFrame([])

        horse_number = np.nan
        money_return = np.nan
        for tr in table.find_all('tr'):
            if '単勝' in tr.text:
                horse_number = int(tr.find_all('td')[0].text)
                money_return = int(tr.find_all('td', attrs={'class':'txt_r'})[0].text.replace(',', ''))
                break
        tmp['race_id'] = [race_id]
        tmp['horse_number'] = [horse_number]
        tmp['money_return'] = [money_return]

        return tmp
