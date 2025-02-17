# CIFAR-10 Classification Project

This repository contains the implementation of a CIFAR-10 image classification project. The project is structured into two phases, each focusing on different aspects of model development and testing. Below is a description of the folder structure and the purpose of each file.

## Folder Structure
```
YourDirectoryID_hw0.zip
|   Phase1 
|   ├── Code
|   |   ├── Wrapper.py           # Code for basic image processing and filter operations.
|   Phase2
|   ├── Code
|   |   ├── Train.py             # Training script for CIFAR-10 classification.
|   |   ├── Test.py              # Testing script to evaluate the trained model.
|   |   ├── Network.py           # Implementation of the neural networks used in the project.
|   |   |   └── HW0Phase1AndPhase2Notebook.ipynb  # Jupyter notebook combining Phase 1 and Phase 2.
├── Report.pdf                   # Detailed project report.
└── README.md                    # This file.
```

## Files Description

### Phase 1
- **Wrapper.py**: Implements basic image processing techniques, including Gaussian, Gabor, and other custom filters. This is a utility script for preprocessing images used in the project.

### Phase 2
- **Train.py**: Contains the training pipeline for CIFAR-10 classification. It includes dataset loading, data augmentation, model initialization, and optimization routines.
- **Test.py**: Script for testing and evaluating the trained models. It generates confusion matrices, accuracy scores, and other evaluation metrics.
- **Network.py**: Houses the implementation of several neural network architectures, such as LeNet, ResNet, DenseNet, ResNeXt, and a custom CIFAR10Model.
- **HW0Phase1AndPhase2Notebook.ipynb**: Kindly refer above folders for code

### Root Files
- **Report.pdf**: A detailed report summarizing the project, including methodologies, results, and analyses.
- **README.md**: This file, providing an overview of the project structure and usage.

## How to Use

### Prerequisites
- Python 3.x
- Required Python packages: `torch`, `torchvision`, `numpy`, `opencv-python`, `matplotlib`, `sklearn`, `seaborn`, and others as specified in the scripts.

### Phase 1
1. Navigate to the `Phase1/Code/` directory.
2. Run `Wrapper.py` to perform image processing tasks, such as generating filter banks and processing input images.

### Phase 2
1. Navigate to the `Phase2/Code/` directory.
2. Train the model:
   ```bash
   python Train.py --NumEpochs 25 --MiniBatchSize 128
   ```
3. Test the model:
   ```bash
   python Test.py --ModelPath ../Checkpoints_dense/ResNext_24model.ckpt
   ```

### Notebook
1.Follow the steps in the notebook to run both Phase 1 and Phase 2 workflows interactively.

## Results
The results of the project, including model accuracy, confusion matrices, and other evaluation metrics, are summarized in the `Report.pdf` file.

## Acknowledgment
This project was guided by **Dr. Nitin J. Sanket** at the **Worcester Polytechnic Institute**, offering a fun and challenging way to learn basic computer vision and machine learning concepts.
