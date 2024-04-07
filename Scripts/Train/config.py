import sklearn.preprocessing as sp
import pickle

## standardscalerの設定値を調整してください

class Dataset_Config(object):
    def __init__(self):

        self.type_categories = ['ダ', '芝']
        self.weather_categories = ['晴', '雨', '小雨', '小雪', '曇', '雪']
        self.condition_categories = ['良', '稍', '重', '不']
        self.sex_categories = ['牡', '牝', 'セ']

        self.type_enc = sp.OneHotEncoder(categories=[self.type_categories], handle_unknown='ignore')
        self.weather_enc = sp.OneHotEncoder(categories=[self.weather_categories], handle_unknown='ignore')
        self.condition_enc = sp.OneHotEncoder(categories=[self.condition_categories], handle_unknown='ignore')
        self.sex_enc = sp.OneHotEncoder(categories=[self.sex_categories], handle_unknown='ignore')

        self.one_hot_columns = ['type', 'weather', 'condition', 'sex']
        self.numerical_columns = ['horse_number', 'weight_carried', 'length', 'age', 'weight', 'weight_difference', 'month', 'velocity']
        self.output_columns = ['velocity']

        self.ven_input_columns = ['horse_number', 'weight_carried', 'length', 'month',
                                  'type_ダ', 'type_芝', 
                                  'weather_晴', 'weather_雨', 'weather_小雨', 'weather_小雪', 'weather_曇', 'weather_雪',
                                  'condition_良', 'condition_稍', 'condition_重', 'condition_不', 
                                  'sex_牡', 'sex_牝', 'sex_セ']
        
        self.ven_numerical_columns = ['horse_number', 'weight_carried', 'length', 'month']
        
        
        # self.sin_input_columns = ['horse_number', 'weight_carried', 'length', 'month', 'age', 'weight', 'weight_difference', 
        #                           'type_ダ', 'type_芝', 
        #                           'weather_晴', 'weather_雨', 'weather_小雨', 'weather_小雪', 'weather_曇', 'weather_雪',
        #                           'condition_良', 'condition_稍', 'condition_重', 'condition_不', 
        #                           'sex_牡', 'sex_牝', 'sex_セ']
        
        self.original_columns = ['勝率', '連対 率', '複勝 率']
        self.sin_jockey_input_columns = ['jockey_win_rate', 'jockey_place_rate', 'jockey_show_rate']
        self.sin_trainer_input_columns = ['trainer_win_rate', 'trainer_place_rate', 'trainer_show_rate']

        self.horse_past_number = 5
        self.sin_numerical_columns = [f'past_{i+1}' for i in range(self.horse_past_number)] + self.sin_jockey_input_columns + self.sin_trainer_input_columns

        self.sin_jockey_input_dict = dict(zip(self.original_columns, self.sin_jockey_input_columns))
        self.sin_trainer_input_dict = dict(zip(self.original_columns, self.sin_trainer_input_columns))

        self.ven_info_path = 'Data/Info/ven_standardscaler.pkl'
        self.sin_info_path = 'Data/Info/sin_standardscaler.pkl'

        self.horse_past_columns = 'horse_last'
        self.horse_folder = 'Data/Horse/'
        self.jockey_folder = 'Data/Jockey/'
        self.trainer_folder = 'Data/Trainer/'

    def ven_fit_std(self, df):
        df['month'] = df['date'].apply(self.get_month)
        df = df.loc[:, self.ven_numerical_columns]
        self.std = sp.StandardScaler()
        self.std.fit(df)

        with open(self.ven_info_path, 'wb') as file:
            pickle.dump(self.std, file)

    def sin_fit_std(self, df):
        df = df.loc[:, self.sin_numerical_columns]
        self.std = sp.StandardScaler()
        self.std.fit(df)

        with open(self.sin_info_path, 'wb') as file:
            pickle.dump(self.std, file)

    def get_month(self, x):
        month = x.month
        return month
    


class Train_Config(object):
    def __init__(self):

        dataset_config = Dataset_Config()
        self.ven_input_dim = len(dataset_config.ven_input_columns)
        self.sin_input_dim = dataset_config.horse_past_number + len(dataset_config.sin_jockey_input_columns) + len(dataset_config.sin_trainer_input_columns)

        self.ven_model_path = 'Model/ven_model.pth'
        self.ven_optimizer_path = 'Model/ven_optimizer.pth'
        self.ven_log_folder = 'Log/VEN/'

        self.sin_model_path = 'Model/sin_model.pth'
        self.sin_optimizer_path = 'Model/sin_optimizer.pth'
        self.sin_log_folder = 'Log/SIN/'