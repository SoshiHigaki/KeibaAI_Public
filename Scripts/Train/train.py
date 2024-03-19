from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import os

from Scripts.Train import model
from Scripts.Train import dataset


class Config(object):
    def __init__(self):
        self.ve_net_input_dim = 21


class Train(object):
    def __init__(self, 
                 device):
        self.config = Config()

        self.device = device

        self.dataset = dataset.Dataset(self.device)

        self.ve_net = model.VelocityEvaluationNetwork(self.config.ve_net_input_dim)
        self.ve_net.to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.ve_net.parameters(), lr=0.01)

        self.batch_size = 2**10
        self.log_folder = 'Log/'

    def train(self, paths, epochs):
        self.ve_net.train()

        for path in tqdm(paths):
            self.dataset.prepare(path)
            data_loader = DataLoader(self.dataset,
                                     batch_size=self.batch_size,
                                     shuffle=True)
            path_list = path.split('/')
            log_path = f'{self.log_folder}/{path_list[-2]}-{path_list[-1]}'
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
            
        return history
                