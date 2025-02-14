# We will implement the network here
"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Colab file can be found at:
    https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute



Code adapted from CMSC733 at the University of Maryland, College Park.
"""

from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def loss_fn(out, labels):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################
    criterion = nn.CrossEntropyLoss()  # Cross Entropy Loss
    loss = criterion(out, labels)
    return loss

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        images, labels = images.cuda(), labels.cuda()
        out = self(images)                  # Generate predictions
        loss = loss_fn(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        images, labels = images.cuda(), labels.cuda()
        out = self(images)                    # Generate predictions
        loss = loss_fn(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'loss': loss.detach(), 'acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'loss': epoch_loss.item(), 'acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], loss: {:.4f}, acc: {:.4f}".format(epoch, result['loss'], result['acc']))




# LeNet********************************************************************************************************************

class LeNet(ImageClassificationBase):
  def __init__(self, InputSize, OutputSize):
      """
      Inputs: 
      InputSize - Size of the Input
      OutputSize - Size of the Output
      """
      super().__init__()
      self.LeNet = nn.Sequential(
        nn.Conv2d(3, 6, kernel_size = 5, padding = 0),
        nn.Tanh(),
        nn.AvgPool2d(2,2),
        nn.Conv2d(6, 16, kernel_size = 5, padding = 0),
        nn.Tanh(),
        nn.AvgPool2d(2,2),
        nn.Conv2d(16, 120, kernel_size = 5, padding = 0),
        nn.Tanh(),
        nn.Flatten(),
        nn.Linear(120, 48),
        nn.Tanh(),
        nn.Linear(48, OutputSize)
      )

      
  def forward(self, xb):
      """
      Input:
      xb is a MiniBatch of the current image
      Outputs:
      out - output of the network
      """
      out = self.LeNet(xb)
     
      return out

# CIFAR10Model*********************************************************************************************************************

class CIFAR10Model(ImageClassificationBase):# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ MY MODEL
    def __init__(self, InputSize = 32, OutputSize=10):
        """
        Inputs: 
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
       
        super(CIFAR10Model, self).__init__()
         ###############################################
        # Fill your network initialization of choice here!
        ###############################################
    
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5_bn = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, OutputSize)

        # self.dropout = nn.Dropout(0.5)
         
    def forward(self, xb):


        """
    #     Input:
    #     xb is a MiniBatch of the current image
    #     Outputs:
    #     out - output of the network
    #     """
              #############################
        #     # Fill your network structure of choice here!
        #     #############################
        xb = F.relu(self.conv1_bn(self.conv1(xb)))
        xb = F.relu(self.conv2_bn(self.conv2(xb)))
        xb = self.pool1(xb)

        xb = F.relu(self.conv3_bn(self.conv3(xb)))
        xb = F.relu(self.conv4_bn(self.conv4(xb)))
        xb = self.pool2(xb)

        xb = F.relu(self.conv5_bn(self.conv5(xb)))
        xb = self.pool3(xb)

        xb = xb.view(xb.size(0), -1)
        xb = F.relu(self.fc1(xb))
        # xb = self.dropout(xb)
        xb = F.relu(self.fc2(xb))
        xb = self.fc3(xb)
        return xb

#


#ResNet******************************************************************************************************************** 

class ResNet(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.res_block_1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.res_block_2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512)
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(), 
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, xb):
        out = self.conv_block_1(xb)
        out = F.relu(self.res_block_1(out) + out)  
        out = self.conv_block_2(out)
        out = F.relu(self.res_block_2(out) + out)  
        out = self.classifier(out)
        return out






# ###################################################################################

# # DenseNetCIFAR10********************************************************************************************************************
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(self._build_layer(in_channels, growth_rate))
            in_channels += growth_rate

    def _build_layer(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate * 4, kernel_size=1, bias=False),  # Bottleneck layer
            nn.BatchNorm2d(growth_rate * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(growth_rate * 4, growth_rate, kernel_size=3, padding=1, bias=False)  # 3x3 Convolution
        )

    def forward(self, xb):
        for layer in self.layers:
            new_features = layer(xb)
            xb = torch.cat([xb, new_features], dim=1)  # Concatenate along the channel axis
        return xb


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),  # 1x1 Convolution
            nn.AvgPool2d(kernel_size=2, stride=2)  # Downsample
        )

    def forward(self, xb):
        return self.transition(xb)


class DenseNetCIFAR10(ImageClassificationBase):
    def __init__(self, growth_rate=12, num_dense_blocks=3, num_layers_per_block=6, num_classes=10):
        super().__init__()
        num_channels = 2 * growth_rate  # Initial number of channels
        self.initial_conv = nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False)  # Initial 3x3 Convolution

        self.blocks = nn.ModuleList()
        for i in range(num_dense_blocks):
            self.blocks.append(DenseBlock(num_channels, growth_rate, num_layers_per_block))
            num_channels += growth_rate * num_layers_per_block  # Update the number of channels
            if i != num_dense_blocks - 1:  # Add a transition layer after every dense block except the last
                out_channels = num_channels // 2
                self.blocks.append(TransitionLayer(num_channels, out_channels))
                num_channels = out_channels

        self.final_bn = nn.BatchNorm2d(num_channels)
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
            nn.Flatten(),
            nn.Linear(num_channels, num_classes)
        )

    def forward(self, xb):
        out = self.initial_conv(xb)
        for block in self.blocks:
            out = block(out)
        out = self.final_bn(out)
        out = self.classifier(out)
        return out




#ResNeXT********************************************************************************************************************


class res_block(nn.Module):
    def __init__(self, in_channels, cardinality, basewidth, stride):
        super().__init__()
        gp_width = cardinality*basewidth               # as described in paper
        self.layers = nn.Sequential(nn.Conv2d(in_channels, gp_width, kernel_size = 1),
                                    nn.BatchNorm2d(gp_width),
                                    nn.Conv2d(gp_width, gp_width, kernel_size = 3, stride = stride, padding = 1, groups = cardinality), #2 groups made(as C=2 in our case)
                                    nn.BatchNorm2d(gp_width),
                                    nn.Conv2d(gp_width, 2*gp_width, kernel_size = 1),
                                    nn.BatchNorm2d(2*gp_width))
        self.add = nn.Sequential(nn.Conv2d(in_channels, 2*gp_width, kernel_size = 1, stride = stride),
                                    nn.BatchNorm2d(2*gp_width))
    
    def forward(self, xb):
        out = self.layers(xb)
        out += self.add(xb)
        out = F.relu(out)
        return out




class ResNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, cardinality):
        super().__init__()
        # Ensure group_width is divisible by cardinality
        group_width = (out_channels // cardinality) * cardinality
        self.conv1 = nn.Conv2d(in_channels, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        self.conv3 = nn.Conv2d(group_width, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, xb):
        out = F.relu(self.bn1(self.conv1(xb)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(xb)
        out = F.relu(out)
        return out


class ResNeXtCIFAR10(ImageClassificationBase):
    def __init__(self, cardinality=8, num_classes=10):
        super().__init__()
        self.in_channels = 64

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)

        # ResNeXt stages
        self.stage1 = self._make_layer(128, 3, stride=1, cardinality=cardinality)
        self.stage2 = self._make_layer(256, 3, stride=2, cardinality=cardinality)
        self.stage3 = self._make_layer(512, 3, stride=2, cardinality=cardinality)

        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride, cardinality):
        layers = []
        for i in range(num_blocks):
            layers.append(ResNeXtBlock(self.in_channels, out_channels, stride if i == 0 else 1, cardinality))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, xb):
        out = F.relu(self.bn1(self.conv1(xb)))
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.global_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
