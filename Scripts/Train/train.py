from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import os

from Scripts.Train import model
from Scripts.Train import dataset

## standardscalerの設定値を調整してください

class Config(object):
    def __init__(self):
        self.ve_net_input_dim = 21

        self.ve_net_model_path = 'Model/ve_net_model.pth'
        self.ve_net_optimizer_path = 'Model/ve_net_optimizer.pth'
        self.ve_log_folder = 'Log/EV/'

        self.si_net_model_path = 'Model/si_net_model.pth'
        self.si_net_optimizer_path = 'Model/si_net_optimizer.pth'
        self.si_log_folder = 'Log/SI/'


class ev_train(object):
    def __init__(self, 
                 device):
        self.config = Config()
        self.device = device

        self.ve_dataset = dataset.ve_dataset(self.device)

        self.ve_net = model.VelocityEvaluationNetwork(self.config.ve_net_input_dim)
        self.ve_net.to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.ve_net.parameters(), lr=0.01)

        self.batch_size = 2**10

        self.ve_net_model_path = self.config.ve_net_model_path
        self.ve_net_optimizer_path = self.config.ve_net_optimizer_path
        self.ve_log_folder = self.config.ve_log_folder

        self.ve_net.load_state_dict(torch.load(self.ve_net_model_path)) if os.path.exists(self.ve_net_model_path) else None
        self.optimizer.load_state_dict(torch.load(self.ve_net_optimizer_path)) if os.path.exists(self.ve_net_optimizer_path) else None

    def train(self, paths, epochs):
        self.ve_net.train()

        for path in tqdm(paths):
            self.ve_dataset.prepare(path)
            data_loader = DataLoader(self.ve_dataset,
                                     batch_size=self.batch_size,
                                     shuffle=True)
            path_list = path.split('/')
            log_path = f'{self.ve_log_folder}/{path_list[-2]}-{path_list[-1]}'
            history = list(pd.read_pickle(log_path)['loss']) if os.path.exists(log_path) else []
            for _ in tqdm(range(epochs), leave=False):
                epoch_loss = 0
                for data in data_loader:
                    input_data, output_data = data['input'], data['output']

                    outputs = self.ve_net(input_data)
                    loss = self.criterion(outputs, output_data)

                    epoch_loss += loss.item()
                    
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                history.append(epoch_loss / len(data_loader))
                
            history_df = pd.DataFrame({'epochs':list(range(1, len(history)+1, 1)), 'loss':history})
            history_df.to_pickle(log_path)

        torch.save(self.ve_net.state_dict(), self.ve_net_model_path)
        torch.save(self.optimizer.state_dict(), self.ve_net_optimizer_path)   


class si_train(object):
    def __init__(self, 
                 device):
        self.config = Config()
        self.device = device

        self.si_dataset = dataset.si_dataset(self.device)

        self.horse_past_number = self.si_dataset.horse_past_number

        self.si_net = model.SpeedIndexNetwork(self.horse_past_number)
        self.si_net.to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.si_net.parameters(), lr=0.01)

        self.batch_size = 2**10

        self.si_net_model_path = self.config.si_net_model_path
        self.si_net_optimizer_path = self.config.si_net_optimizer_path
        self.si_log_folder = self.config.si_log_folder

        self.si_net.load_state_dict(torch.load(self.si_net_model_path)) if os.path.exists(self.si_net_model_path) else None
        self.optimizer.load_state_dict(torch.load(self.si_net_optimizer_path)) if os.path.exists(self.si_net_optimizer_path) else None

        self.ve_net = model.VelocityEvaluationNetwork(self.config.ve_net_input_dim)
        self.ve_net.to(self.device)
        self.ve_net.load_state_dict(torch.load(self.config.ve_net_model_path)) if os.path.exists(self.config.ve_net_model_path) else None
        self.ve_net.eval()

    def train(self, paths, epochs):
        self.si_net.train()

        for path in tqdm(paths):
            self.si_dataset.prepare(path)
            data_loader = DataLoader(self.si_dataset,
                                     batch_size=self.batch_size,
                                     shuffle=True)
            path_list = path.split('/')
            log_path = f'{self.si_log_folder}/{path_list[-2]}-{path_list[-1]}'
            history = list(pd.read_pickle(log_path)['loss']) if os.path.exists(log_path) else []
            for _ in tqdm(range(epochs), leave=False):
                epoch_loss = 0
                for data in data_loader:
                    race_input, race_output, horse_input, horse_output = data['race_input'], data['race_output'], data['horse_input'], data['horse_output']

                    race_model_output = self.ve_net(race_input)
                    race_si = race_output - race_model_output

                    horse_input = torch.reshape(horse_input, (-1, self.config.ve_net_input_dim))
                    horse_output = torch.reshape(horse_output, (-1, 1))
 
                    horse_model_output = self.ve_net(horse_input)

                    horse_past_si = horse_output - horse_model_output
                    horse_past_si = torch.where(torch.isnan(horse_past_si), torch.tensor(0.), horse_past_si)
                    horse_past_si = torch.reshape(horse_past_si, (-1, self.horse_past_number))

                    si_model_output = self.si_net(horse_past_si)

                    loss = self.criterion(race_si, si_model_output)

                    epoch_loss += loss.item()
                    
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                history.append(epoch_loss / len(data_loader))
                
            history_df = pd.DataFrame({'epochs':list(range(1, len(history)+1, 1)), 'loss':history})
            history_df.to_pickle(log_path)

        torch.save(self.si_net.state_dict(), self.si_net_model_path)
        torch.save(self.optimizer.state_dict(), self.si_net_optimizer_path)   