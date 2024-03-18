import torch
import sklearn.preprocessing as sp
import pickle
import pandas as pd

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.type_categories = ['ダ', '芝']
        self.weather_categories = ['晴', '雨', '小雨', '小雪', '曇', '雪']
        self.condition_categories = ['良', '稍', '重', '不']
        self.sex_categories = ['牡', '牝', 'セ']

        self.type_enc = sp.OneHotEncoder(categories=[self.type_categories], handle_unknown='ignore')
        self.weather_enc = sp.OneHotEncoder(categories=[self.weather_categories], handle_unknown='ignore')
        self.condition_enc = sp.OneHotEncoder(categories=[self.condition_categories], handle_unknown='ignore')
        self.sex_enc = sp.OneHotEncoder(categories=[self.sex_categories], handle_unknown='ignore')

        self.info_path = 'Data/Info/standardscaler.pkl'

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
        self.std = sp.StandardScaler()
        self.std.fit(df)

        with open(self.info_path, 'wb') as file:
            pickle.dump(self.std, file)

    def load_std(self):
        with open(self.info_path, 'rb') as file:
            self.std = pickle.load(file)

    def transform_std(self, df):
        data = self.std.transform(df)
        return data