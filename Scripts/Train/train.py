from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import os
from torch.optim.lr_scheduler import LambdaLR

from Scripts.Train import config 
from Scripts.Train import dataset
from Scripts.Train import model
from Scripts.Train import early_stopping



class VEN_Train(object):
    def __init__(self, 
                 device):
        
        self.batch_size = 2**10
        
        self.config = config.Train_Config()
        self.device = device

        self.train_ven_dataset = dataset.VEN_Dataset(self.device)
        self.test_ven_dataset = dataset.VEN_Dataset(self.device)

        self.ven_model = model.VelocityEvaluationNetwork(self.config.ven_input_dim, 1)
        self.ven_model.to(self.device)

        self.initial_lr = 0.01
        self.final_lr = 0.001
        self.ven_criterion = nn.MSELoss()
        self.ven_optimizer = torch.optim.Adam(self.ven_model.parameters(), lr=self.initial_lr, amsgrad=True)

        self.ven_model_path = self.config.ven_model_path
        self.ven_optimizer_path = self.config.ven_optimizer_path
        self.ven_log_folder = self.config.ven_log_folder

        self.ven_model.load_state_dict(torch.load(self.ven_model_path)) if os.path.exists(self.ven_model_path) else None
        self.ven_optimizer.load_state_dict(torch.load(self.ven_optimizer_path)) if os.path.exists(self.ven_optimizer_path) else None

        self.early_stopping = early_stopping.EarlyStopping(threshold=5.0*10**(-3))

    def train(self, train_paths, test_paths, epochs):
        self.ven_model.train()

        for train_path, test_path in tqdm(zip(train_paths, test_paths), total=len(train_paths)):

            lambda_lr = lambda epoch: 1 - (1 - self.final_lr / self.initial_lr) * (epoch / (epochs-1))
            scheduler = LambdaLR(self.ven_optimizer, lr_lambda=lambda_lr)

            self.train_ven_dataset.prepare(train_path)
            train_data_loader = DataLoader(self.train_ven_dataset,
                                            batch_size=self.batch_size,
                                            shuffle=True)
            
            self.test_ven_dataset.prepare(test_path)
            test_data_loader = DataLoader(self.test_ven_dataset,
                                            batch_size=self.batch_size,
                                            shuffle=True)
            
            path_list = train_path.split('/')
            log_path = f'{self.ven_log_folder}/{path_list[-2]}-{path_list[-1]}'
            root, old_extension = os.path.splitext(log_path)
            log_path = root + '.csv'

            self.early_stopping.initialize()
            for epoch in tqdm(range(epochs), leave=False, desc=f'{train_path}'):
                epoch_loss = 0
                self.ven_model.train()
                for data in train_data_loader:
                    input_data, output_data = data['input'], data['output']

                    outputs = self.ven_model(input_data)
                    loss = self.ven_criterion(outputs, output_data)

                    epoch_loss += loss.item()
                    
                    self.ven_optimizer.zero_grad()
                    loss.backward()
                    self.ven_optimizer.step()

                train_loss = epoch_loss / len(train_data_loader)

                epoch_loss = 0
                self.ven_model.eval()
                with torch.no_grad():
                    for data in test_data_loader:
                        input_data, output_data = data['input'], data['output']

                        outputs = self.ven_model(input_data)
                        loss = self.ven_criterion(outputs, output_data)

                        epoch_loss += loss.item()
                        
                test_loss = epoch_loss / len(test_data_loader)

                self.save_log(log_path,
                              train_loss,
                              test_loss)

                if self.early_stopping(test_loss):
                    print('Early Stop')
                    break

                torch.save(self.ven_model.state_dict(), self.ven_model_path)
                torch.save(self.ven_optimizer.state_dict(), self.ven_optimizer_path) 

                scheduler.step()

    def save_log(self,
                 log_path,
                 
                 train_loss,
                 test_loss):
        tmp_log = pd.DataFrame([])
        tmp_log['train_loss'] = [train_loss]
        tmp_log['test_loss'] = [test_loss]

        mode = 'a' if os.path.exists(log_path) else 'w'
        header = False if os.path.exists(log_path) else True

        tmp_log.to_csv(log_path, mode=mode, index=False, header=header)


class SIN_Train(object):
    def __init__(self, 
                 device):
        
        self.batch_size = 2**10
        
        self.config = config.Train_Config()
        self.device = device

        self.train_sin_dataset = dataset.SIN_Dataset(self.device, self.batch_size)
        self.test_sin_dataset = dataset.SIN_Dataset(self.device, self.batch_size)

        self.sin_model = model.SpeedIndexNetwork(self.config.sin_input_dim, 1)
        self.sin_model.to(self.device)

        self.initial_lr = 0.01
        self.final_lr = 0.001
        self.sin_criterion = nn.MSELoss()
        self.sin_optimizer = torch.optim.Adam(self.sin_model.parameters(), lr=self.initial_lr, amsgrad=True)

        self.sin_model_path = self.config.sin_model_path
        self.sin_optimizer_path = self.config.sin_optimizer_path
        self.sin_log_folder = self.config.sin_log_folder

        self.sin_model.load_state_dict(torch.load(self.sin_model_path)) if os.path.exists(self.sin_model_path) else None
        self.sin_optimizer.load_state_dict(torch.load(self.sin_optimizer_path)) if os.path.exists(self.sin_optimizer_path) else None

        self.early_stopping = early_stopping.EarlyStopping(threshold=5.0*10**(-3))

    def train(self, train_paths, test_paths, epochs):
        self.sin_model.train()

        for train_path, test_path in tqdm(zip(train_paths, test_paths), total=len(train_paths)):

            lambda_lr = lambda epoch: 1 - (1 - self.final_lr / self.initial_lr) * (epoch / (epochs-1))
            scheduler = LambdaLR(self.sin_optimizer, lr_lambda=lambda_lr)

            self.train_sin_dataset.prepare(train_path)
            train_data_loader = DataLoader(self.train_sin_dataset,
                                            batch_size=self.batch_size,
                                            shuffle=True)
            
            self.test_sin_dataset.prepare(test_path)
            test_data_loader = DataLoader(self.test_sin_dataset,
                                            batch_size=self.batch_size,
                                            shuffle=True)
            
            path_list = train_path.split('/')
            log_path = f'{self.sin_log_folder}/{path_list[-2]}-{path_list[-1]}'
            root, old_extension = os.path.splitext(log_path)
            log_path = root + '.csv'

            self.early_stopping.initialize()
            for epoch in tqdm(range(epochs), leave=False, desc=f'{train_path}'):
                epoch_loss = 0
                self.sin_model.train()
                for data in train_data_loader:
                    input_data, output_data = data['input'], data['output']

                    outputs = self.sin_model(input_data)
                    loss = self.sin_criterion(outputs, output_data)

                    epoch_loss += loss.item()
                    
                    self.sin_optimizer.zero_grad()
                    loss.backward()
                    self.sin_optimizer.step()

                train_loss = epoch_loss / len(train_data_loader)

                epoch_loss = 0
                self.sin_model.eval()
                with torch.no_grad():
                    for data in test_data_loader:
                        input_data, output_data = data['input'], data['output']

                        outputs = self.sin_model(input_data)
                        loss = self.sin_criterion(outputs, output_data)

                        epoch_loss += loss.item()
                        
                test_loss = epoch_loss / len(test_data_loader)

                self.save_log(log_path,
                              train_loss,
                              test_loss)

                if self.early_stopping(test_loss):
                    print('Early Stop')
                    break

                torch.save(self.sin_model.state_dict(), self.sin_model_path)
                torch.save(self.sin_optimizer.state_dict(), self.sin_optimizer_path) 

                scheduler.step()

    def save_log(self,
                 log_path,
                 
                 train_loss,
                 test_loss):
        tmp_log = pd.DataFrame([])
        tmp_log['train_loss'] = [train_loss]
        tmp_log['test_loss'] = [test_loss]

        mode = 'a' if os.path.exists(log_path) else 'w'
        header = False if os.path.exists(log_path) else True

        tmp_log.to_csv(log_path, mode=mode, index=False, header=header)