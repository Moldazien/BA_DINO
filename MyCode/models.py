import torch
import torch.nn as nn
import torch.nn.functional as F


##########################################
#
# einfachstes netz. duch svm nicht mehr sinnvoll
#
##########################################
class SimpleNet(nn.Module):

    def __init__(self):
      super(SimpleNet, self).__init__()

      self.conv1 = nn.Conv2d(in_channels=12, out_channels=1, kernel_size=1, stride=1)
      self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1)

    def forward(self, x):
      x = self.conv1(x)
      x = self.conv2(x)
      x = F.relu(x)
      return x

##########################################
#
# cnn mit tiefe von 4. ist ein bisschen unsinnig aufgebaut
#
##########################################
class SimpleCNN(nn.Module):

    def __init__(self):
      super(SimpleCNN, self).__init__()

      self.conv1 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
      self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
      self.conv3 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
      self.conv4 = nn.Conv2d(in_channels=12, out_channels=1, kernel_size=1, stride=1, padding=0, padding_mode='reflect')


    def forward(self, x):
      x = self.conv1(x)
      x = F.relu(x)
      x = self.conv2(x)
      x = F.relu(x)
      x = self.conv3(x)
      x = F.relu(x)
      x = self.conv4(x)
      #x = F.relu(x)

      return x     


##########################################
#
# cnn mit tiefe von 6. letzte schicht macht so nicht wirklich sinn
# hier verwendet mit sigmoid layer als letzes. bei inferenz nicht vergessen
#
##########################################
class BiggerSimpleCNN(nn.Module):

    def __init__(self):
      super(BiggerSimpleCNN, self).__init__()

      self.conv1 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
      self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
      self.conv3 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
      self.conv4 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
      self.conv5 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
      self.conv6 = nn.Conv2d(in_channels=12, out_channels=1, kernel_size=1, stride=1, padding=0, padding_mode='reflect')


    def forward(self, x):
      x = self.conv1(x)
      x = F.relu(x)
      x = self.conv2(x)
      x = F.relu(x)
      x = self.conv3(x)
      x = F.relu(x)
      x = self.conv4(x)
      x = F.relu(x)
      x = self.conv5(x)
      x = F.relu(x)
      x = self.conv6(x)

      return x


##########################################
#
# cnn mit tiefe von 8. funktioniert nicht. -> lernt einfach nicht. loss verändert sich nicht. auch nach mehr als 60 epochen
#
##########################################     