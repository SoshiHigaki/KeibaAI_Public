from Scripts.Train import train
from Scripts.Train import model
from Scripts.Train import get_today_data
from Scripts.Train import dataset

from torch.utils.data import DataLoader
import os
import pandas as pd
import numpy as np
import torch


class Predict(object):
    def __init__(self,
                 device):
        self.device = device

        self.get_today_data = get_today_data.TodaysData()
        self.si_dataset = dataset.si_dataset(self.device)

        self.config = train.Config()

        self.horse_past_number = self.si_dataset.horse_past_number

        self.si_net = model.SpeedIndexNetwork(self.config.ve_net_input_dim + self.horse_past_number)
        self.si_net.to(self.device)
        self.si_net.load_state_dict(torch.load(self.config.si_net_model_path)) if os.path.exists(self.config.si_net_model_path) else None
        self.si_net.eval()

        self.ve_net = model.VelocityEvaluationNetwork(self.config.ve_net_input_dim)
        self.ve_net.to(self.device)
        self.ve_net.load_state_dict(torch.load(self.config.ve_net_model_path)) if os.path.exists(self.config.ve_net_model_path) else None
        self.ve_net.eval()

    def main(self, url):
        self.get_today_data.main(url)

        self.si_dataset.prepare(self.get_today_data.save_path)
        data_loader = DataLoader(self.si_dataset,
                                 batch_size=len(self.si_dataset.race_df),
                                 shuffle=True)

        for data in data_loader:
            race_input, race_output, horse_input, horse_output = data['race_input'], data['race_output'], data['horse_input'], data['horse_output']

            horse_input = torch.reshape(horse_input, (-1, self.config.ve_net_input_dim))
            horse_output = torch.reshape(horse_output, (-1, 1))
 
            horse_model_output = self.ve_net(horse_input)

            horse_past_si = horse_output - horse_model_output
            horse_past_si = torch.where(torch.isnan(horse_past_si), torch.tensor(0.), horse_past_si)
            horse_past_si = torch.reshape(horse_past_si, (-1, self.horse_past_number))

            input_data = torch.cat((race_input, horse_past_si), dim=1)

            si_model_output = self.si_net(input_data)

        result = pd.DataFrame()
        result['SpeedIndex'] = pd.Series(si_model_output.detach().numpy().flatten())
        result['Horse Name'] = np.loadtxt(self.get_today_data.horse_names_path, dtype=str)
        result['Horse Number'] = np.loadtxt(self.get_today_data.horse_numbers_path, dtype=int)

        result = result.sort_values(by='SpeedIndex', ascending=False)

        return result