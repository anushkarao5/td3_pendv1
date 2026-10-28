import numpy as np
import torch
import torch.nn as nn
import torch.optim
import random

class ActorNetwork(nn.Module):
  '''
  Goal: Map pendulem state to torque action
  Inputs: STATE: [cos(theta), sin(theta), angular_velocity]
  Output: Torque in [-action_magnitude to action_magnitude]
  '''
  def __init__(self,action_magnitude=2):
    super().__init__()
    self.action_magnitude=action_magnitude

    self.layer1=nn.Linear(3,64)
    self.layer2=nn.Linear(64,128)
    self.layer3=nn.Linear(128,1)

  def forward(self,x):
    x=torch.relu(self.layer1(x))
    x=torch.relu(self.layer2(x))

    x=self.action_magnitude*torch.tanh(self.layer3(x))
    return x

class CriticNetwork(nn.Module):
  '''
  Goal: Estimates q value from from state and action
  Inputs: STATE + ACTION [cost(theta), sin(theta), angular_velcoity, action]
  Output: q value (estimated discounted return)
  '''
  def __init__(self):
    super().__init__()
    self.layer1=nn.Linear(4,64)
    self.layer2=nn.Linear(64,128)
    self.layer3=nn.Linear(128,1)

  def forward(self,x):
    x=torch.relu(self.layer1(x))
    x=torch.relu(self.layer2(x))
    x=self.layer3(x)
    return x