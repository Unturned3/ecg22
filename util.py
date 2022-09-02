
from time import sleep
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

# data handling
import pandas as pd

# misc
import os


import os
import time
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts,
                                      StepLR,
                                      ExponentialLR)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score

class Config:
    csv_path = ''
    seed = 123456
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    attn_state_path = 'DS/attn.pth'
    lstm_state_path = 'DS/lstm.pth'
    cnn_state_path = 'DS/cnn.pth'

    attn_logs = 'DS/attn.csv'
    lstm_logs = 'DS/lstm.csv'
    cnn_logs = 'DS/cnn.csv'

    train_csv_path = ''
    test_csv_path = ''

config = Config()

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class ECGDataset(Dataset):

    def __init__(self, df):
        self.df = df
        self.data_columns = self.df.columns[:-2].tolist()

    def __getitem__(self, idx):
        signal: pd.Series = self.df.loc[idx, self.data_columns].astype('float32')

        # Seems like creating a tensor from a list of
        # numpy.ndarrays is extremely slow, compared to
        # using .to_numpy() before converting it to a tensor.

        # Original:
        # signal = torch.FloatTensor([signal.values])

        # Optimized:
        signal = np.expand_dims(signal.to_numpy(), 0)
        signal = torch.Tensor(signal)

        target = torch.LongTensor(np.array(self.df.loc[idx, 'class']))
        return signal, target

    def __len__(self):
        return len(self.df)

def get_dataloader(phase: str, batch_size: int = 96) -> DataLoader:
    df = pd.read_csv(config.train_csv_path)
    train_df, val_df = train_test_split(
        df, test_size=0.15, random_state=config.seed, stratify=df['label']
    )
    train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)
    df = train_df if phase == 'train' else val_df
    dataset = ECGDataset(df)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4)
    return dataloader


class _MITBIH_TEST_DS(Dataset):
    
    def __init__(self, csv_dir, csv_file):
        self.csv = pd.read_csv(os.path.join(csv_dir, csv_file), header=None).to_numpy()
    
    def __len__(self) -> int:
        return len(self.csv)
    
    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:

        if not isinstance(idx, int):
            raise TypeError('type(idx) should be int')

        if torch.is_tensor(idx):    # what does this do?
            idx = idx.tolist()

        lbl = np.eye(5)[int(self.csv[idx, 187])]
        ecg = self.csv[idx, :186]

        ecg = torch.from_numpy(ecg)
        #ecg = torch.unsqueeze(ecg, dim=0)
        lbl = torch.from_numpy(lbl)
        #lbl = torch.unsqueeze(lbl, dim=0)

        return (ecg.float(), lbl.float())

def _plot_ecg(ecg: np.ndarray, title: str) -> None:

    duration = 1.488    # 186 samples at 125Hz

    x = np.linspace(0, duration, len(ecg))

    fig, ax = plt.subplots()

    fig.suptitle(title)
    fig.set_size_inches(12, 2)
    fig.set_dpi(80)

    ax.set_xlabel('seconds')
    ax.set_ylabel('mV')
    ax.set_xlim(left=0, right=duration+0.2)

    ax.plot(x, ecg)
    plt.show()


class Meter:
    def __init__(self, n_classes=5):
        self.metrics = {}
        self.confusion = torch.zeros((n_classes, n_classes))

    def update(self, x, y, loss):
        x = np.argmax(x.detach().cpu().numpy(), axis=1)
        y = y.detach().cpu().numpy()
        self.metrics['loss'] += loss
        self.metrics['accuracy'] += accuracy_score(x, y)
        self.metrics['f1'] += f1_score(x, y, average='macro')
        self.metrics['precision'] += precision_score(
            x, y, average='macro', zero_division=1)
        self.metrics['recall'] += recall_score(x,
                                               y, average='macro', zero_division=1)

        self._compute_cm(x, y)

    def _compute_cm(self, x, y):
        for prob, target in zip(x, y):
            if prob == target:
                self.confusion[target][target] += 1
            else:
                self.confusion[target][prob] += 1

    def init_metrics(self):
        self.metrics['loss'] = 0
        self.metrics['accuracy'] = 0
        self.metrics['f1'] = 0
        self.metrics['precision'] = 0
        self.metrics['recall'] = 0

    def get_metrics(self):
        return self.metrics

    def get_confusion_matrix(self):
        return self.confusion


class Trainer:
    def __init__(self, net, lr, batch_size, num_epochs):
        self.net = net.to(config.device)
        self.num_epochs = num_epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(self.net.parameters(), lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=5e-6)
        self.best_loss = float('inf')
        self.phases = ['train', 'val']
        self.dataloaders = {
            phase: get_dataloader(phase, batch_size) for phase in self.phases
        }
        self.train_df_logs = pd.DataFrame()
        self.val_df_logs = pd.DataFrame()
    
    def _train_epoch(self, phase):
        print(f"{phase} mode | time: {time.strftime('%H:%M:%S')}")
        
        self.net.train() if phase == 'train' else self.net.eval()
        meter = Meter()
        meter.init_metrics()
        
        for i, (data, target) in enumerate(self.dataloaders[phase]):
            data = data.to(config.device)
            target = target.to(config.device)
            
            output = self.net(data)
            loss = self.criterion(output, target)
                        
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            meter.update(output, target, loss.item())
        
        metrics = meter.get_metrics()
        metrics = {k:v / i for k, v in metrics.items()}
        df_logs = pd.DataFrame([metrics])
        confusion_matrix = meter.get_confusion_matrix()
        
        if phase == 'train':
            self.train_df_logs = pd.concat([self.train_df_logs, df_logs], axis=0)
        else:
            self.val_df_logs = pd.concat([self.val_df_logs, df_logs], axis=0)
        
        # show logs
        print('{}: {}, {}: {}, {}: {}, {}: {}, {}: {}'
              .format(*(x for kv in metrics.items() for x in kv))
             )
        fig, ax = plt.subplots(figsize=(5, 5))
        cm_ = ax.imshow(confusion_matrix, cmap='hot')
        ax.set_title('Confusion matrix', fontsize=15)
        ax.set_xlabel('Actual', fontsize=13)
        ax.set_ylabel('Predicted', fontsize=13)
        plt.colorbar(cm_)
        plt.show()
        
        return loss
    
    def run(self):
        for epoch in range(self.num_epochs):
            self._train_epoch(phase='train')
            with torch.no_grad():
                val_loss = self._train_epoch(phase='val')
                self.scheduler.step()
            
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                print('\nNew checkpoint\n')
                self.best_loss = val_loss
                torch.save(self.net.state_dict(), f"best_model_epoc{epoch}.pth")
            #clear_output()
