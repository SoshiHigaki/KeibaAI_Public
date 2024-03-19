import torch
import sklearn.preprocessing as sp
import pickle
import pandas as pd

import warnings
warnings.simplefilter('ignore', FutureWarning)

class ve_dataset(torch.utils.data.Dataset):
    def __init__(self, device):

        self.device = device

        self.type_categories = ['ダ', '芝']
        self.weather_categories = ['晴', '雨', '小雨', '小雪', '曇', '雪']
        self.condition_categories = ['良', '稍', '重', '不']
        self.sex_categories = ['牡', '牝', 'セ']

        self.type_enc = sp.OneHotEncoder(categories=[self.type_categories], handle_unknown='ignore')
        self.weather_enc = sp.OneHotEncoder(categories=[self.weather_categories], handle_unknown='ignore')
        self.condition_enc = sp.OneHotEncoder(categories=[self.condition_categories], handle_unknown='ignore')
        self.sex_enc = sp.OneHotEncoder(categories=[self.sex_categories], handle_unknown='ignore')

        self.info_path = 'Data/Info/standardscaler.pkl'

    def prepare(self, path):
        df = pd.read_pickle(path)
        df = df.dropna()
        df = df.reset_index(drop=True)
        
        df = df.drop(['race_id', 'horse_id', 'jockey_id', 'trainer_id', 'date'], axis=1)

        one_hot_columns = ['type', 'weather', 'condition', 'sex']
        one_hot_df = df.loc[:, one_hot_columns]
        
        numerical_columns = ['horse_number', 'weight_carried', 'length', 'age', 'weight', 'weight_difference', 'velocity']
        numerical_df = df.loc[:, numerical_columns]

        one_hot_data = self.one_hot(one_hot_df)

        self.load_std()
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
        df = df.drop(['type', 'weather', 'condition', 'sex'], axis=1)

        return df
    
    def fit_std(self, df):
        df = df.loc[:, ['horse_number', 'weight_carried', 'length', 'age', 'weight', 'weight_difference', 'velocity']]
        self.std = sp.StandardScaler()
        self.std.fit(df)

        with open(self.info_path, 'wb') as file:
            pickle.dump(self.std, file)

    def load_std(self):
        with open(self.info_path, 'rb') as file:
            self.std = pickle.load(file)

    def transform_std(self, df):
        data = self.std.transform(df)
        data = pd.DataFrame(data, columns=df.columns)
        return data
    
    def __getitem__(self, i):
        input_data = torch.tensor(self.df.loc[i, ['horse_number', 'weight_carried', 'length', 'age', 'weight',
                                        'weight_difference', 'type_ダ', 'type_芝', 'weather_晴',
                                        'weather_雨', 'weather_小雨', 'weather_小雪', 'weather_曇', 'weather_雪',
                                        'condition_良', 'condition_稍', 'condition_重', 'condition_不', 'sex_牡',
                                        'sex_牝', 'sex_セ']], dtype=torch.float32).to(self.device)
        
        output_data = torch.tensor(self.df.loc[i, ['velocity']], dtype=torch.float32).to(self.device)

        data = {'input':input_data, 'output':output_data}

        return data

    def __len__(self): 
        return len(self.df)