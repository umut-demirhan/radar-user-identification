# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 21:38:20 2021

@author: Umt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
    
class NetGraphGeneral(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc11 = nn.Linear(1, 16)
        self.fc12 = nn.Linear(16, 32)
        self.fc13 = nn.Linear(32, 64)
        
        
        self.fc21 = nn.Linear(3, 16)
        self.fc22 = nn.Linear(16, 32)
        self.fc23 = nn.Linear(32, 64)
        
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)
        self.fc7 = nn.Linear(32, 16)
        
        self.fc8 = nn.Linear(16, 1)
        
    def forward(self, x_obj, x_beam):
        
        x_beam = F.relu(self.fc11(x_beam))
        x_beam = F.relu(self.fc12(x_beam))
        x_beam = F.relu(self.fc13(x_beam))
        
        x_obj = F.relu(self.fc21(x_obj))
        x_obj = F.relu(self.fc22(x_obj))
        x_obj = F.relu(self.fc23(x_obj))

        z = torch.cat([x_beam, x_obj], dim=1) # Repeat the beams for each object of same sample
        z = F.relu(self.fc4(z))
        z = F.relu(self.fc5(z))
        z = F.relu(self.fc6(z))
        z = F.relu(self.fc7(z))
        z = torch.sigmoid(self.fc8(z))
        
        return z
