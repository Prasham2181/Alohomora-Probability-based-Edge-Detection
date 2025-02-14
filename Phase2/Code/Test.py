# !/usr/bin/env python3

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


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

import cv2
import os
import sys
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
import torchvision
import torchvision.transforms as tf
import argparse
import shutil
import string
import math as m
from sklearn.metrics import confusion_matrix
from tqdm.notebook import tqdm
import seaborn as sns
import torch

from Network.Network import CIFAR10Model,DenseNetCIFAR10,LeNet, ResNet, ResNeXtCIFAR10
from Misc.MiscUtils import *
from Misc.DataUtils import *


# Don't generate pyc codes
sys.dont_write_bytecode = True

def SetupAll():
    """
    Outputs:
    ImageSize - Size of the Image
    """   
    # Image Input Shape
    ImageSize = [32, 32, 3]

    return ImageSize

def StandardizeInputs(Img):
    ##########################################################################
    # Add any standardization or cropping/resizing if used in Training here!
    ##########################################################################

    #standardization has been in code main()
    return Img
    
def ReadImages(Img):
    """
    Outputs:
    I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """    
    I1 = Img
    
    if(I1 is None):
        # OpenCV returns empty list if image is not read! 
        print('ERROR: Image I1 cannot be read')
        sys.exit()
        
    I1S = StandardizeInputs(np.float32(I1))

    I1Combined = np.expand_dims(I1S, axis=0)

    return I1Combined, I1
                

def Accuracy(Pred, GT):
    """
    Inputs: 
    Pred are the predicted labels
    GT are the ground truth labels
    Outputs:
    Accuracy in percentage
    """
    return (np.sum(np.array(Pred)==np.array(GT))*100.0/len(Pred))

def ReadLabels(LabelsPathTest, LabelsPathPred):
    if(not (os.path.isfile(LabelsPathTest))):
        print('ERROR: Test Labels do not exist in '+LabelsPathTest)
        sys.exit()
    else:
        LabelTest = open(LabelsPathTest, 'r')
        LabelTest = LabelTest.read()
        LabelTest = map(float, LabelTest.split())

    if(not (os.path.isfile(LabelsPathPred))):
        print('ERROR: Pred Labels do not exist in '+LabelsPathPred)
        sys.exit()
    else:
        LabelPred = open(LabelsPathPred, 'r')
        LabelPred = LabelPred.read()
        LabelPred = map(float, LabelPred.split())
        
    return LabelTest, LabelPred

def ConfusionMatrix(LabelsTrue, LabelsPred):
    """
    LabelsTrue - True labels
    LabelsPred - Predicted labels
    """

    # Convert labels to lists
    LabelsTrue, LabelsPred = list(LabelsTrue), list(LabelsPred)

    # Compute confusion matrix
    cm = confusion_matrix(y_true=LabelsTrue, y_pred=LabelsPred)

    # Print the confusion matrix as text
    print("Confusion Matrix:")
    for i in range(len(cm)):
        print(str(cm[i, :]) + f" ({i})")

    # Print the class numbers for reference
    class_numbers = [f" ({i})" for i in range(len(cm))]
    print("".join(class_numbers))

    # Calculate and print accuracy
    accuracy = Accuracy(LabelsPred, LabelsTrue)
    print(f"Accuracy: {accuracy:.2f}%")

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(len(cm)), yticklabels=range(len(cm)))
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix\nAccuracy: {accuracy:.2f}%")
    plt.show()

def TestOperation(ImageSize, ModelPath, TestSet, LabelsPathPred):
    """
    Inputs: 
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    TestSet - The test dataset
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to /content/data/TxtFiles/PredOut.txt
    """
    
    
    # Predict output with forward pass, MiniBatchSize for Test is 1
    # model = CIFAR10Model(InputSize=3*32*32,OutputSize=10)  # For CIFAR10 --- MY MODEL 
    # model = DenseNetCIFAR10()
    # model = LeNet(3,10) 

    # model = ResNet(3,10) 
    model = ResNeXtCIFAR10(cardinality=8, num_classes=10)    
    
    CheckPoint = torch.load(ModelPath)
    # Optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-4)
    model.load_state_dict(CheckPoint['model_state_dict'])
    # Optimizer.load_state_dict(CheckPoint['optimizer_state_dict'])
    model.eval()
    # print('Number of parameters in this model are %d ' % len(model.state_dict().items()))
    total_params = sum(param.numel() for param in model.parameters())
    print('Parameters', total_params)
    label_list = []
    pred_list = []
    OutSaveT = open(LabelsPathPred, 'w')

    for count in tqdm(range(len(TestSet))): 
        Img, Label = TestSet[count]
        label_list.append(Label)
        # Img, ImgOrg = ReadImages(Img)
        PredT = model(Img.unsqueeze(0))
        # print(PredT.shape)
        PredT = torch.argmax(PredT, dim =1).item()
        pred_list.append(PredT)

        OutSaveT.write(str(PredT)+'\n')
    OutSaveT.close()
    return label_list, pred_list

       
def main():
    """
    Inputs: 
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    # Parser.add_argument('--ModelPath', dest='ModelPath', default='/home/aa/144model.ckpt', help='Path to load latest model from, Default:ModelPath')
    Parser.add_argument('--ModelPath', dest='ModelPath', default='../Checkpoints_dense/ResNext_24model.ckpt', help='Path to load latest model from, Default:ModelPath')    
    Parser.add_argument('--LabelsPath', dest='LabelsPath', default='./TxtFiles/LabelsTest.txt', help='Path of labels file, Default:./TxtFiles/LabelsTest.txt')
    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    LabelsPath = Args.LabelsPath
    
    transform_test = tf.Compose([tf.ToTensor(),
                                tf.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                ])
    
    TestSet = torchvision.datasets.CIFAR10(root='data/', train=False, download = True, transform=transform_test)
    TrainSet = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_test)

    # Setup all needed parameters including file reading
    ImageSize = SetupAll()

    # Define PlaceHolder variables for Predicted output
    LabelsPathPred = './TxtFiles/PredOut.txt' # Path to save predicted labels

    label_list, pred_list = TestOperation(ImageSize, ModelPath, TestSet, LabelsPathPred)

    # Plot Confusion Matrix
    LabelsTrue, LabelsPred = ReadLabels(LabelsPath, LabelsPathPred)
    # ConfusionMatrix(LabelsTrue, LabelsPred)
    ConfusionMatrix(label_list, pred_list)
     
if __name__ == '__main__':
    main()
