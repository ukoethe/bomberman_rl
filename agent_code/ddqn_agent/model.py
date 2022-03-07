import torch
from torch import nn
from torch import autograd

class CNN(nn.Module):
    def __init__(self, available_actions_count, input_shape) -> None:
        """defining the CNN architecture for the Agent to decide with
        Args:
            available_actions_count (int): numbers of actions the agent can take in a single step
        """
        super().__init__()
        
        self.input_shape = (1, ) + input_shape
        self.convultions = nn.Sequential(
            # Conv Layer 1
            nn.Conv2d(1, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            
            # Conv Layer 2
            nn.Conv2d(8, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            
            # Conv Layer 3
            nn.Conv2d(8, 8, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            
            # Conv Layer 4
            nn.Conv2d(8, 16, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()     
        )

        self.softmax = nn.Softmax()

        self.logsoftmax = nn.LogSoftmax()

        # return convultions, softmax, logsoftmax

class DDQN(nn.Module):

    def __init__(self) -> None:
        pass

    def model_architecture(self) -> Model:
        pass