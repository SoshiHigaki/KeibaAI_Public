import numpy as np

class EarlyStopping:
    def __init__(self, patience=10, threshold=1.0*10**(-2)):
        self.patience = patience
        self.threshold = threshold
        
        self.counter = 0
        self.min_loss = None
        self.early_stop = False
        self.previous_loss = np.Inf 

    def __call__(self, loss):
        if abs(loss-self.previous_loss)<self.threshold: #学習が収束している場合
            if self.min_loss is None:
                self.min_loss = loss

            if loss > self.min_loss: #最小を更新できなかった場合
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.min_loss = loss #最小損失を更新できた場合
                self.counter = 0

        else:
            self.counter = 0

        self.previous_loss = loss
        return self.early_stop

    def initialize(self):
        self.counter = 0
        self.min_loss = None
        self.early_stop = False
        self.previous_loss = np.Inf 