import torch
import torch.nn as nn


class L_net(nn.Module):
    def __init__(self, num=64):
        super(L_net, self).__init__()
        self.L_net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, num, 3, 1, 0),
            nn.ReLU(),               
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(), 
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),               
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),   
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, 1, 3, 1, 0),
        )

    def forward(self, input):
        return torch.sigmoid(self.L_net(input))


class R_net(nn.Module):
    def __init__(self, num=64):
        super(R_net, self).__init__()

        self.R_net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, num, 3, 1, 0),
            nn.ReLU(), 
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),               
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),               
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),            
            nn.ReLU(),   
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, 3, 3, 1, 0),
        )

    def forward(self, input):
        return torch.sigmoid(self.R_net(input))

class N_net(nn.Module):
    def __init__(self, num=64):
        super(N_net, self).__init__()
        self.N_net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, num, 3, 1, 0),
            nn.ReLU(), 
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),               
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),               
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),            
            nn.ReLU(),   
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, 3, 3, 1, 0),
        )

    def forward(self, input):
        return torch.sigmoid(self.N_net(input))


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()        
        self.L_net = L_net(num=64)
        self.R_net = R_net(num=64)
        self.N_net = N_net(num=64)        

    def forward(self, input):
        x = self.N_net(input)
        L = self.L_net(x)
        R = self.R_net(x)
        return L, R, x
