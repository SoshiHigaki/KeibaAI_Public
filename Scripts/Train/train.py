from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd

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

    def train(self, paths, epochs):
        self.ve_net.train()

        history = pd.DataFrame([])
        for path in tqdm(paths):
            self.dataset.prepare(path)
            data_loader = DataLoader(self.dataset,
                                     batch_size=self.batch_size,
                                     shuffle=True)
            
            tmp_history = []
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

                tmp_history.append(epoch_loss / len(data_loader))
                

            history[path] = tmp_history

        return history
                




        