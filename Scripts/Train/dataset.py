import torch
import sklearn.preprocessing as sp
import pickle
import pandas as pd
import numpy as np

import warnings
warnings.simplefilter('ignore', FutureWarning)

class Config(object):
    def __init__(self):
        self.type_categories = ['ダ', '芝']
        self.weather_categories = ['晴', '雨', '小雨', '小雪', '曇', '雪']
        self.condition_categories = ['良', '稍', '重', '不']
        self.sex_categories = ['牡', '牝', 'セ']

        self.type_enc = sp.OneHotEncoder(categories=[self.type_categories], handle_unknown='ignore')
        self.weather_enc = sp.OneHotEncoder(categories=[self.weather_categories], handle_unknown='ignore')
        self.condition_enc = sp.OneHotEncoder(categories=[self.condition_categories], handle_unknown='ignore')
        self.sex_enc = sp.OneHotEncoder(categories=[self.sex_categories], handle_unknown='ignore')

        self.drop_columns = ['race_id', 'horse_id', 'jockey_id', 'trainer_id', 'date']
        self.one_hot_columns = ['type', 'weather', 'condition', 'sex']
        self.numerical_columns = ['horse_number', 'weight_carried', 'length', 'age', 'weight', 'weight_difference', 'velocity']

        self.input_columns = ['horse_number', 'weight_carried', 'length', 'age', 'weight',
                                'weight_difference', 'type_ダ', 'type_芝', 'weather_晴',
                                'weather_雨', 'weather_小雨', 'weather_小雪', 'weather_曇', 'weather_雪',
                                'condition_良', 'condition_稍', 'condition_重', 'condition_不', 'sex_牡',
                                'sex_牝', 'sex_セ']
        self.output_columns = ['velocity']

        self.info_path = 'Data/Info/standardscaler.pkl'

        self.horse_past_number = 5
        self.horse_folder = 'Data/Horse/'

    def fit_std(self, df):
        df = df.loc[:, ['horse_number', 'weight_carried', 'length', 'age', 'weight', 'weight_difference', 'velocity']]
        self.std = sp.StandardScaler()
        self.std.fit(df)

        with open(self.info_path, 'wb') as file:
            pickle.dump(self.std, file)

class ve_dataset(torch.utils.data.Dataset):
    def __init__(self, device):

        self.device = device

        self.config = Config()
        self.type_enc = self.config.type_enc
        self.weather_enc = self.config.weather_enc
        self.condition_enc = self.config.condition_enc
        self.sex_enc = self.config.sex_enc

        self.drop_columns = self.config.drop_columns
        self.one_hot_columns = self.config.one_hot_columns
        self.numerical_columns = self.config.numerical_columns

        self.input_columns = self.config.input_columns
        self.output_columns = self.config.output_columns

        self.info_path = self.config.info_path

        self.load_std()

    def prepare(self, path):
        df = pd.read_pickle(path)
        df = df.dropna()
        df = df.reset_index(drop=True)
        
        df = df.drop(self.drop_columns, axis=1)

        one_hot_df = df.loc[:, self.one_hot_columns]
        numerical_df = df.loc[:, self.numerical_columns]

        one_hot_data = self.one_hot(one_hot_df)
        numerical_data = self.transform_std(numerical_df)

        self.df = pd.concat([numerical_data, one_hot_data], axis=1)

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
        with open(self.info_path, 'rb') as file:
            self.std = pickle.load(file)

    def transform_std(self, df):
        data = self.std.transform(df)
        data = pd.DataFrame(data, columns=df.columns)
        return data
    
    def __getitem__(self, i):
        input_data = torch.tensor(self.df.loc[i, self.input_columns], dtype=torch.float32).to(self.device)
        
        output_data = torch.tensor(self.df.loc[i, self.output_columns], dtype=torch.float32).to(self.device)

        data = {'input':input_data, 'output':output_data}

        return data

    def __len__(self): 
        return len(self.df)
    

class si_dataset(torch.utils.data.Dataset):
    def __init__(self, device):

        self.device = device

        self.config = Config()
        self.type_enc = self.config.type_enc
        self.weather_enc = self.config.weather_enc
        self.condition_enc = self.config.condition_enc
        self.sex_enc = self.config.sex_enc

        self.drop_columns = self.config.drop_columns
        self.one_hot_columns = self.config.one_hot_columns
        self.numerical_columns = self.config.numerical_columns

        self.input_columns = self.config.input_columns
        self.output_columns = self.config.output_columns

        self.info_path = self.config.info_path

        self.horse_past_number = self.config.horse_past_number
        self.horse_folder = self.config.horse_folder

        self.load_std()

    def prepare(self, path):
        dates, horse_ids = self.prepare_race(path)
        self.prepare_horse(dates, horse_ids)

    def prepare_race(self, path):
        race_df = pd.read_pickle(path)
        race_df = race_df.dropna()
        race_df = race_df.reset_index(drop=True)

        dates = list(race_df['date'])
        horse_ids = list(race_df['horse_id'])
        
        race_df = race_df.drop(self.drop_columns, axis=1)

        one_hot_df = race_df.loc[:, self.one_hot_columns]
        numerical_df = race_df.loc[:, self.numerical_columns]

        one_hot_data = self.one_hot(one_hot_df)
        numerical_data = self.transform_std(numerical_df)

        self.race_df = pd.concat([numerical_data, one_hot_data], axis=1)

        return dates, horse_ids
    
    def prepare_horse(self, date, horse_ids):
        self.horse_data = []
        for d, id in zip(date, horse_ids):
            path = f'{self.horse_folder}/{id}.pkl'
            df = pd.read_pickle(path)
            df = df.loc[df['date'] < d].reset_index(drop=True)
            df = self.adjust_dataframe_length(df)
            df = df.reset_index(drop=True)

            df = df.drop(['date'], axis=1)

            one_hot_df = df.loc[:, self.one_hot_columns]
            
            numerical_df = df.loc[:, self.numerical_columns]

            one_hot_data = self.one_hot(one_hot_df)
            numerical_data = self.transform_std(numerical_df)

            df = pd.concat([numerical_data, one_hot_data], axis=1)

            self.horse_data.append(df)


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

    def load_std(self):
        with open(self.info_path, 'rb') as file:
            self.std = pickle.load(file)

    def transform_std(self, df):
        data = self.std.transform(df)
        data = pd.DataFrame(data, columns=df.columns)
        return data
    
    def __getitem__(self, i):
        race_input = torch.tensor(self.race_df.loc[i, self.input_columns], dtype=torch.float32).to(self.device)
        race_output = torch.tensor(self.race_df.loc[i, self.output_columns], dtype=torch.float32).to(self.device)

        horse_input = torch.tensor(self.horse_data[i].loc[:, self.input_columns].values, dtype=torch.float32).to(self.device)
        horse_output = torch.tensor(self.horse_data[i].loc[:, self.output_columns].values, dtype=torch.float32).to(self.device)

        data = {'race_input':race_input, 'race_output':race_output, 'horse_input':horse_input, 'horse_output':horse_output}

        return data

    def __len__(self): 
        return len(self.race_df)