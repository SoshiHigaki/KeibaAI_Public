import torch
from tqdm.notebook import tqdm
import pickle
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader

from Scripts.Train import config 
from Scripts.Train import model

import warnings
warnings.simplefilter('ignore', FutureWarning)

class VEN_Dataset(torch.utils.data.Dataset):
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

    def prepare(self, path):

        df = pd.read_pickle(path)
        df = df.dropna()
        df = df.reset_index(drop=True)
        df['month'] = df['date'].apply(self.get_month)
        
        one_hot_df = df.loc[:, self.one_hot_columns]
        numerical_df = df.loc[:, self.numerical_columns]

        one_hot_data = self.one_hot(one_hot_df)
        numerical_data = self.transform_std(numerical_df)

        self.df = pd.concat([numerical_data, one_hot_data], axis=1)
        self.df = self.df.astype(float)

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
        input_data = torch.tensor(self.df.loc[i, self.input_columns].values, dtype=torch.float32).to(self.device)
        output_data = torch.tensor(self.df.loc[i, self.output_columns].values, dtype=torch.float32).to(self.device)

        data = {'input':input_data, 'output':output_data}

        return data

    def __len__(self): 
        return len(self.df)
    

class SIN_Dataset(torch.utils.data.Dataset):
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
        df = df.dropna()
        df = df.reset_index(drop=True)

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
        self.df = self.df.dropna()
        self.df = self.df.reset_index(drop=True)

        self.load_std()
        self.df[self.input_columns] = self.std.transform(self.df[self.input_columns])

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
         
    def load_std(self):
        with open(self.sin_info_path, 'rb') as file:
            self.std = pickle.load(file)

    def get_year(self, x):
        return x.year
        
    def horse_fillna(self, row):
        row_mean = row.mean()
        return row.fillna(row_mean)

    def __getitem__(self, i):
        input_data = torch.tensor(self.df.loc[i, self.input_columns].values, dtype=torch.float32).to(self.device)
        output_data = torch.tensor(self.df.loc[i, self.output_columns].values, dtype=torch.float32).to(self.device) 

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
        self.df = self.df.astype(float)

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