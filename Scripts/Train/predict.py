import torch
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader
import pickle

import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
from datetime import datetime
from urllib.parse import urlparse, parse_qs

from Scripts.Prepare import get_horse_data
from Scripts.Prepare import get_jockey_data
from Scripts.Prepare import get_trainer_data

from Scripts.Train import config
from Scripts.Train import dataset
from Scripts.Train import model

class Predict(object):
    def __init__(self, device):
        self.config = config.Train_Config()
        self.device = device

        self.batch_size = 100
        self.dataset = Predict_Dataset(self.device, self.batch_size)

        self.sin_model = model.SpeedIndexNetwork(self.config.sin_input_dim, 1)
        self.sin_model.to(self.device)

        self.sin_model_path = self.config.sin_model_path
        self.sin_model.load_state_dict(torch.load(self.sin_model_path)) if os.path.exists(self.sin_model_path) else None

        self.today_path = 'Data/today.pkl'
        self.todaysdata = TodaysData()

    def predict(self, url):
        self.todaysdata.main(url)

        delete_df = self.dataset.prepare(self.today_path)
        data_loader = DataLoader(self.dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False)
        
        horse_number = list(self.dataset.df['horse_number'])
        horse_name = list(self.dataset.df['horse_name'])

        self.sin_model.eval()
        with torch.no_grad():
            for data in data_loader:
                input_data, output_data = data['input'], data['output']
                outputs = self.sin_model(input_data)

        result = pd.DataFrame([])
        result['horse number'] = horse_number
        result['horse name'] = horse_name
        result['speedindex'] = list(outputs.squeeze().detach().cpu().numpy())

        mean_value = result['speedindex'].mean()
        std_dev = result['speedindex'].std()
        speedindex = np.array(result.loc[:, ['speedindex']])
        result['Deviation'] = (speedindex - mean_value) / std_dev * 10 + 50

        result = result.sort_values(by='Deviation', ascending=False)
        result = result.reset_index(drop=True)

        delete_df = delete_df.reset_index(drop=True)

        return result, delete_df

class TodaysData(object):
    def __init__(self):
        self.get_horse_data = get_horse_data.Get_Horse_Data()
        self.get_jockey_data = get_jockey_data.Get_Jockey_Data()
        self.get_trainer_data = get_trainer_data.Get_Trainer_Data()
        
        self.save_path = 'Data/today.pkl'
    
    def main(self, url):
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        race_id = query_params.get('race_id')[0]

        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        df = self.get_table(soup)
        info = self.get_race_info(soup)
        date = self.get_date(soup)

        df = self.clean(df)
        df = self.concat(df, info, date, race_id)
        df = self.select_columns(df)

        df = self.set_type(df)

        horse_ids = list(df['horse_id'])
        self.get_horse_data.main(horse_ids)

        year = date.year
        self.get_jockey_data.main(year, year)
        self.get_trainer_data.main(year, year)

        df.to_pickle(self.save_path)


    def get_table(self, soup):
        
        table = soup.find('div', attrs={'class':'RaceTableArea'})
        start_index = 2
        if table == None:
            table = soup.find('div', attrs={'class':'ResultTableWrap'})
            start_index = 1
                        
        headers = []
        data = []

        for th in table.find_all('th'):
            headers.append(th.text.strip())

        horse_ids = []
        jockey_ids = []
        trainer_ids = []
        for tr in table.find_all('tr')[start_index:]:
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

        try:
            headers.remove('お気に入り馬')
        except:
            pass

        df = pd.DataFrame(data, columns=headers)
        df['horse_id'] = horse_ids
        df['jockey_id'] = jockey_ids
        df['trainer_id'] = trainer_ids

        df = df.loc[:, ['馬名', 'horse_id', 'jockey_id', 'trainer_id', '馬番', '性齢', '斤量', '馬体重(増減)']]
        df.columns = ['horse_name', 'horse_id', 'jockey_id', 'trainer_id', 'horse_number', 'sex_and_age', 'weight_carried', 'horse_weight']

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

        df = df.fillna(0)
        df = df.replace('', 0)
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
        df = df.loc[:, ['horse_name', 'race_id', 'horse_id', 'jockey_id', 'trainer_id', 'date', 
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

class Predict_Dataset(torch.utils.data.Dataset):
    def __init__(self, device, batchsize):

        self.device = device
        self.batchsize = batchsize

        self.config = config.Dataset_Config()

        self.horse_dataset = Horse_Dataset(self.device)
        self.race_dataset = Race_Dataset(self.device)

        self.jockey_input_columns = self.config.sin_jockey_input_columns
        self.jockey_folder = self.config.jockey_folder
        self.jockey_rename_dict = self.config.sin_jockey_input_dict

        self.trainer_input_columns = self.config.sin_trainer_input_columns
        self.trainer_folder = self.config.trainer_folder
        self.trainer_rename_dict = self.config.sin_trainer_input_dict

        self.sin_info_path = self.config.sin_info_path

        self.horse_input_columns = [f'past_{i+1}' for i in range(self.horse_dataset.horse_past_number)]
        self.input_columns = [f'past_{i+1}' for i in range(self.horse_dataset.horse_past_number)] + self.jockey_input_columns + self.trainer_input_columns
        self.output_columns = ['race']

        self.train_config = config.Train_Config()
        self.ven_model = model.VelocityEvaluationNetwork(len(self.config.ven_input_columns), 1)
        self.ven_model.to(self.device)
        self.ven_model.load_state_dict(torch.load(self.train_config.ven_model_path)) if os.path.exists(self.train_config.ven_model_path) else None

    def prepare(self, path):
        df = pd.read_pickle(path)
        df = df.reset_index(drop=True)

        horse_numbers = list(df['horse_number'])
        horse_names = list(df['horse_name'])

        df = self.jockey_data(df)
        df = self.trainer_data(df)
        jockey_df = df.loc[:, self.jockey_input_columns]
        trainer_df = df.loc[:, self.trainer_input_columns]

        self.race_dataset.prepare(df)
        self.horse_dataset.prepare(df)

        race_data_loader = DataLoader(self.race_dataset,
                                      batch_size=self.batchsize,
                                      shuffle=False)

        horse_data_loader = DataLoader(self.horse_dataset,
                                       batch_size=self.batchsize,
                                       shuffle=False)
        
        self.ven_model = model.VelocityEvaluationNetwork(len(self.config.ven_input_columns), 1)
        self.ven_model.to(self.device)
        self.ven_model.load_state_dict(torch.load(self.train_config.ven_model_path)) if os.path.exists(self.train_config.ven_model_path) else None
        
        self.ven_model.eval()
        race_tensor = torch.tensor([]).to(self.device)
        with torch.no_grad():
            for data in race_data_loader:
                input_data, output_data = data['input'], data['output']

                output_data = torch.squeeze(output_data)
                outputs = torch.squeeze(self.ven_model(input_data))

                race_si = output_data - outputs
                race_tensor = torch.cat((race_tensor, race_si), dim=0)

        race_tensor = race_tensor.detach().cpu().numpy()
        race_df = pd.DataFrame(race_tensor)
        race_df.columns = self.output_columns

        horse_tensor = torch.tensor([]).to(self.device)
        with torch.no_grad():
            for data in horse_data_loader:
                input_data, output_data = data['input'], data['output']

                output_data = torch.squeeze(output_data)
                outputs = torch.squeeze(self.ven_model(input_data))

                horse_si = output_data - outputs
                horse_tensor = torch.cat((horse_tensor, horse_si), dim=0)
    
        horse_tensor = torch.reshape(horse_tensor, (-1, self.horse_dataset.horse_past_number))
        horse_tensor = horse_tensor.detach().cpu().numpy()

        horse_df = pd.DataFrame(horse_tensor).apply(self.horse_fillna, axis=1)
        horse_df.columns = self.horse_input_columns

        self.df = pd.concat([race_df, horse_df, jockey_df, trainer_df], axis=1)
        self.df['horse_number'] = horse_numbers
        self.df['horse_name'] = horse_names

        deleted_rows = df[df.isna().any(axis=1)].copy()
        self.df = self.df.dropna()
        self.df = self.df.reset_index(drop=True)

        self.load_std()
        self.df[self.input_columns] = self.std.transform(self.df[self.input_columns])

        return deleted_rows

    def jockey_data(self, df):
        year = list(df['date'].apply(self.get_year))[0]
        jockey_df = pd.read_pickle(self.jockey_folder + f'{year-1}.pkl')

        df = pd.merge(df, jockey_df, on='jockey_id', how='left')
        df = df.rename(columns=self.jockey_rename_dict)
        df = df.fillna(0)

        return df
    
    def trainer_data(self, df):
        year = list(df['date'].apply(self.get_year))[0]
        trainer_df = pd.read_pickle(self.trainer_folder + f'{year-1}.pkl')

        df = pd.merge(df, trainer_df, on='trainer_id', how='left')
        df = df.rename(columns=self.trainer_rename_dict)
        df = df.fillna(0)

        return df
         
    def get_year(self, x):
        return x.year
        
    def load_std(self):
        with open(self.sin_info_path, 'rb') as file:
            self.std = pickle.load(file)

    def horse_fillna(self, row):
        row_mean = row.mean()
        return row.fillna(row_mean)

    def __getitem__(self, i):
        input_data = torch.tensor(self.df.loc[i, self.input_columns].values.astype(float), dtype=torch.float32).to(self.device)
        output_data = torch.tensor(self.df.loc[i, self.output_columns].values.astype(float), dtype=torch.float32).to(self.device)

        data = {'input':input_data, 'output':output_data}

        return data

    def __len__(self): 
        return len(self.df)
    

class Race_Dataset(object):
    def __init__(self, device):
        self.device = device
        
        self.config = config.Dataset_Config()
        
        self.type_enc = self.config.type_enc
        self.weather_enc = self.config.weather_enc
        self.condition_enc = self.config.condition_enc
        self.sex_enc = self.config.sex_enc

        self.one_hot_columns = self.config.one_hot_columns
        self.numerical_columns = self.config.numerical_columns

        self.ven_info_path = self.config.ven_info_path
        self.ven_numerical_columns = self.config.ven_numerical_columns

        self.input_columns = self.config.ven_input_columns
        self.output_columns = self.config.output_columns

    def prepare(self, df):
        self.df = df
        df['month'] = df['date'].apply(self.get_month)

        one_hot_df = self.df.loc[:, self.one_hot_columns]
        numerical_df = self.df.loc[:, self.numerical_columns]

        one_hot_data = self.one_hot(one_hot_df)
        numerical_data = self.transform_std(numerical_df)

        self.df = pd.concat([numerical_data, one_hot_data], axis=1)
        self.df[self.input_columns] = self.df[self.input_columns].astype(float)

        self.load_std()
        self.df[self.ven_numerical_columns] = self.std.transform(self.df[self.ven_numerical_columns])

    def one_hot(self, df):
        type_data = self.type_enc.fit_transform(df[['type']])
        type_df = pd.DataFrame(type_data.toarray(), columns=self.type_enc.get_feature_names_out(['type']))

        weather_data = self.weather_enc.fit_transform(df[['weather']])
        weather_df = pd.DataFrame(weather_data.toarray(), columns=self.weather_enc.get_feature_names_out(['weather']))

        condition_data = self.condition_enc.fit_transform(df[['condition']])
        condition_df = pd.DataFrame(condition_data.toarray(), columns=self.condition_enc.get_feature_names_out(['condition']))

        sex_data = self.sex_enc.fit_transform(df[['sex']])
        sex_df = pd.DataFrame(sex_data.toarray(), columns=self.sex_enc.get_feature_names_out(['sex']))

        df = pd.concat([df, type_df, weather_df, condition_df, sex_df], axis=1)
        df = df.drop(self.one_hot_columns, axis=1)

        return df
    
    def load_std(self):
        with open(self.ven_info_path, 'rb') as file:
            self.std = pickle.load(file)

    def transform_std(self, df):
        # data = self.std.transform(df)
        # data = pd.DataFrame(data, columns=df.columns)
        data = df
        return data
    
    def get_month(self, x):
        month = x.month
        return month
    
    def __getitem__(self, i):

        input_data = torch.tensor(self.df.loc[i, self.input_columns], dtype=torch.float32).to(self.device)
        output_data = torch.tensor(self.df.loc[i, self.output_columns], dtype=torch.float32).to(self.device)

        data = {'input':input_data, 'output':output_data}

        return data

    def __len__(self): 
        return len(self.df)
    

class Horse_Dataset(object):
    def __init__(self, device):
        self.device = device
        
        self.config = config.Dataset_Config()
        
        self.type_enc = self.config.type_enc
        self.weather_enc = self.config.weather_enc
        self.condition_enc = self.config.condition_enc
        self.sex_enc = self.config.sex_enc

        self.one_hot_columns = self.config.one_hot_columns
        self.numerical_columns = self.config.numerical_columns

        self.input_columns = self.config.ven_input_columns
        self.output_columns = self.config.output_columns

        self.horse_past_number = self.config.horse_past_number
        self.horse_folder = self.config.horse_folder

    def prepare(self, df):  
        self.df = df
        self.make_horse_df()


    def make_horse_df(self):
        self.horse_df = self.df.apply(self.get_past_race, axis=1)
        self.horse_df = pd.concat(list(self.horse_df), ignore_index=True).reset_index(drop=True)

        one_hot_df = self.horse_df.loc[:, self.one_hot_columns]
        numerical_df = self.horse_df.loc[:, self.numerical_columns]

        one_hot_data = self.one_hot(one_hot_df)
        numerical_data = self.transform_std(numerical_df)

        self.horse_df = pd.concat([numerical_data, one_hot_data], axis=1)
        self.horse_df = self.horse_df.astype(float)

    def get_past_race(self, row):
        horse_id = row['horse_id']
        date = row['date']

        path = f'{self.horse_folder}/{horse_id}.pkl'
        df = pd.read_pickle(path)
        df['month'] = df['date'].apply(self.get_month)

        df = df.loc[df['date'] < date].reset_index(drop=True)
        df = self.adjust_dataframe_length(df)

        df = df.reset_index(drop=True)
        return df


    def get_month(self, x):
        month = x.month
        return month
    
    def adjust_dataframe_length(self, df):
        current_length = len(df)

        if current_length > self.config.horse_past_number:
            df = df.iloc[:self.config.horse_past_number]  # nより大きい場合は最初のn行を残す
        elif current_length < self.config.horse_past_number:
            nan_rows = pd.DataFrame([np.nan] * len(df.columns)).T
            nan_rows.columns = df.columns
            df = pd.concat([df, pd.concat([nan_rows] * (self.config.horse_past_number - current_length), ignore_index=True)])  # nより小さい場合はNaNで埋める

        return df
    
    def one_hot(self, df):
        type_data = self.type_enc.fit_transform(df[['type']])
        type_df = pd.DataFrame(type_data.toarray(), columns=self.type_enc.get_feature_names_out(['type']))

        weather_data = self.weather_enc.fit_transform(df[['weather']])
        weather_df = pd.DataFrame(weather_data.toarray(), columns=self.weather_enc.get_feature_names_out(['weather']))

        condition_data = self.condition_enc.fit_transform(df[['condition']])
        condition_df = pd.DataFrame(condition_data.toarray(), columns=self.condition_enc.get_feature_names_out(['condition']))

        sex_data = self.sex_enc.fit_transform(df[['sex']])
        sex_df = pd.DataFrame(sex_data.toarray(), columns=self.sex_enc.get_feature_names_out(['sex']))

        df = pd.concat([df, type_df, weather_df, condition_df, sex_df], axis=1)
        df = df.drop(self.one_hot_columns, axis=1)

        return df
    
    def transform_std(self, df):
        # data = self.std.transform(df)
        # data = pd.DataFrame(data, columns=df.columns)
        data = df
        return data
    
    def __getitem__(self, i):

        input_data = torch.tensor(self.horse_df.loc[i, self.input_columns], dtype=torch.float32).to(self.device)
        output_data = torch.tensor(self.horse_df.loc[i, self.output_columns], dtype=torch.float32).to(self.device)

        data = {'input':input_data, 'output':output_data}

        return data

    def __len__(self): 
        return len(self.horse_df)