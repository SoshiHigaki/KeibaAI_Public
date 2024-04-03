import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm.notebook import tqdm
import numpy as np
import os

from Scripts.Train import config
from Scripts.Train import model
from Scripts.Simulation import dataset

import warnings
warnings.simplefilter('ignore', FutureWarning)

class Simulation(object):
    def __init__(self, 
                 device):
        
        self.batch_size = 2**10
        
        self.config = config.Train_Config()
        self.device = device

        self.sin_dataset = dataset.Simulation_Dataset(self.device, self.batch_size)

        self.sin_model = model.SpeedIndexNetwork(self.config.sin_input_dim, 1)
        self.sin_model.to(self.device)

        self.sin_model_path = self.config.sin_model_path
        self.sin_optimizer_path = self.config.sin_optimizer_path
        self.sin_log_folder = self.config.sin_log_folder

        self.sin_model.load_state_dict(torch.load(self.sin_model_path)) if os.path.exists(self.sin_model_path) else None

        self.threshold_deviations = list(range(0, 100, 1))

    def simulate(self, race_paths, return_paths):
        results = []

        for race_path, return_path in tqdm(zip(race_paths, return_paths), total=len(race_paths)):

            horse_si = []
            horse_numbers = []
            race_ids = []

            self.sin_dataset.prepare(race_path)
            data_loader = DataLoader(self.sin_dataset,
                                     batch_size=self.batch_size,
                                     shuffle=False)
            
            horse_numbers.extend(list(self.sin_dataset.df['horse_number']))
            race_ids.extend(list(self.sin_dataset.df['race_id']))
            
            self.sin_model.eval()
            with torch.no_grad():
                for data in data_loader:
                    input_data, output_data = data['input'], data['output']

                    outputs = self.sin_model(input_data)
                    outputs = list(outputs.squeeze().detach().cpu().numpy())

                    horse_si.extend(outputs)

            df = pd.DataFrame([])
            df['race_id'] = race_ids
            df['horse_number'] = horse_numbers
            df['speedindex'] = horse_si

            result = self.calc_return(df, return_path)
            results.append(result)

        return results

    def calc_return(self, result, return_path):
        return_df = pd.read_pickle(return_path)
        return_df['race_id'] = return_df['race_id'].astype('int64').astype(str)
        race_ids = list(set(list(result['race_id'])))

        thresholds = []
        moneys = []
        race_numbers = []
        for t in self.threshold_deviations:
            money = 0
            race_counter = 0
            for id in race_ids:
                tmp = result[result['race_id']==id]
                tmp = tmp.reset_index(drop=True)

                mean_value = tmp['speedindex'].mean()
                std_dev = tmp['speedindex'].std()
                speedindex = np.array(tmp.loc[:, ['speedindex']])
                tmp['Deviation'] = (speedindex - mean_value) / std_dev * 10 + 50

                outstanding_horse = list(tmp[tmp['Deviation']>=t]['horse_number'])
                outstanding_horse = [int(x) for x in outstanding_horse]

                win_horse = int(list(return_df[return_df['race_id']==id]['horse_number'])[0])
                money_return = int(list(return_df[return_df['race_id']==id]['money_return'])[0])

                if win_horse in outstanding_horse:
                    money += money_return
                
                money -= 100 * len(outstanding_horse)

                if len(outstanding_horse) > 0:
                    race_counter += 1

            thresholds.append(t)
            moneys.append(money)
            race_numbers.append(race_counter)

        df = pd.DataFrame([])
        df['threshold'] = thresholds
        df['return'] = moneys
        df['number of race'] = race_numbers

        return df

        


# import torch
# from torch.utils.data import DataLoader
# import pandas as pd
# from tqdm.notebook import tqdm
# import numpy as np
# import os

# from Scripts.Train import config
# from Scripts.Train import model
# from Scripts.Simulation import dataset

# import warnings
# warnings.simplefilter('ignore', FutureWarning)

# class Simulation(object):
#     def __init__(self, 
#                  device):
        
#         self.batch_size = 2**10
        
#         self.config = config.Train_Config()
#         self.device = device

#         self.sin_dataset = dataset.Simulation_Dataset(self.device, self.batch_size)

#         self.sin_model = model.SpeedIndexNetwork(self.config.sin_input_dim, 1)
#         self.sin_model.to(self.device)

#         self.sin_model_path = self.config.sin_model_path
#         self.sin_optimizer_path = self.config.sin_optimizer_path
#         self.sin_log_folder = self.config.sin_log_folder

#         self.sin_model.load_state_dict(torch.load(self.sin_model_path)) if os.path.exists(self.sin_model_path) else None

#     def simulate(self, race_paths, return_paths, threshold_deviation):
#         self.threshold_deviation = threshold_deviation
#         moneys = []
#         race_numbers = []

#         for race_path, return_path in tqdm(zip(race_paths, return_paths), total=len(race_paths)):

#             horse_si = []
#             horse_numbers = []
#             race_ids = []

#             self.sin_dataset.prepare(race_path)
#             data_loader = DataLoader(self.sin_dataset,
#                                      batch_size=self.batch_size,
#                                      shuffle=False)
            
#             horse_numbers.extend(list(self.sin_dataset.df['horse_number']))
#             race_ids.extend(list(self.sin_dataset.df['race_id']))
            
#             self.sin_model.eval()
#             with torch.no_grad():
#                 for data in data_loader:
#                     input_data, output_data = data['input'], data['output']

#                     outputs = self.sin_model(input_data)
#                     outputs = list(outputs.squeeze().detach().cpu().numpy())

#                     horse_si.extend(outputs)

#             df = pd.DataFrame([])
#             df['race_id'] = race_ids
#             df['horse_number'] = horse_numbers
#             df['speedindex'] = horse_si

#             money, race_number = self.calc_return(df, return_path)

#             moneys.append(money)
#             race_numbers.append(race_number)

#         result = pd.DataFrame([])
#         result['data'] = race_paths
#         result['number of race'] = race_numbers
#         result['return'] = moneys

#         return result

#     def calc_return(self, result, return_path):
#         return_df = pd.read_pickle(return_path)
#         return_df['race_id'] = return_df['race_id'].astype('int64').astype(str)
#         race_ids = list(set(list(result['race_id'])))

#         money = 0
#         race_counter = 0
#         for id in tqdm(race_ids):
#             tmp = result[result['race_id']==id]
#             tmp = tmp.reset_index(drop=True)

#             mean_value = tmp['speedindex'].mean()
#             std_dev = tmp['speedindex'].std()
#             speedindex = np.array(tmp.loc[:, ['speedindex']])
#             tmp['Deviation'] = (speedindex - mean_value) / std_dev * 10 + 50

#             outstanding_horse = list(tmp[tmp['Deviation']>=self.threshold_deviation]['horse_number'])
#             outstanding_horse = [int(x) for x in outstanding_horse]

#             win_horse = int(list(return_df[return_df['race_id']==id]['horse_number'])[0])
#             money_return = int(list(return_df[return_df['race_id']==id]['money_return'])[0])

#             if win_horse in outstanding_horse:
#                 money += money_return
            
#             money -= 100 * len(outstanding_horse)

#             if len(outstanding_horse) > 0:
#                 race_counter += 1

#         return money, race_counter