# Probabilistic Boundary Detection and Improving Convolutional Networks

### *RBE549: Computer Vision - [Worcester Polytechnic Institute](https://www.wpi.edu/), Spring 2025*

## Project Overview
This project implements advanced computer vision techniques across two main phases:
1. A probabilistic boundary detection algorithm that improves upon traditional edge detection methods
2. Implementation and optimization of various convolutional neural network architectures

For detailed project specifications, please refer to the [course project page](https://rbe549.github.io/spring2025/hw/hw0/).

## Requirements
- Python 3.8+
- PyTorch
- NumPy
- OpenCV
- scikit-learn
- matplotlib
- tqdm

To install all dependencies:
```bash
pip install -r requirements.txt
```

## Phase 1: Shake My Boundary

### Overview
This phase implements a sophisticated boundary detection algorithm using multiple filter banks and gradient analyses. The approach combines texture, brightness, and color information to create a robust probabilistic boundary detection system.

### Key Features
- Multiple filter bank implementations (DoG, Gabor, HD, LM)
- Multi-channel analysis (texture, brightness, color)
- Gradient computation and combination
- Probabilistic boundary detection

### Steps to Run
1. Ensure your images are in the "BSDS500" directory
2. Run the wrapper script:
```bash
python Wrapper.py
```
All outputs will be automatically saved to the "Outputs" directory.

### Results Visualization

#### Input Image
<p align="left">
  <img src="Images/originalimage_1.jpg" alt="Original Image" style="width: 250px;"/>
</p>

#### Generated Filter Banks
<p align="center">
  <table>
    <tr>
      <td> <img src="Phase_1_media/DoG.png" alt="DoG Filters" style="width: 250px;"/> </td>
      <td> <img src="Phase_1_media/Gabor.png" alt="Gabor Filters" style="width: 250px;"/> </td>
      <td> <img src="Phase_1_media/HDMask.png" alt="HD Masks" style="width: 250px;"/> </td>
      <td> <img src="Phase_1_media/LM.png" alt="LM Filters" style="width: 250px;"/> </td>
    </tr>
    <tr>
      <td align="center">DoG Filters</td>
      <td align="center">Gabor Filters</td>
      <td align="center">HD Masks</td>
      <td align="center">LM Filters</td>
    </tr>
  </table>
</p>

#### Feature Maps and Gradients
<p align="center">
  <table>
    <tr>
      <td> <img src="Phase 2 media_output/img_1/TextonMap_1.png" alt="Texton Map" style="width: 250px;"/> </td>
      <td> <img src="Phase 2 media_output/img_1/BrightnessMap_1.png" alt="Brightness Map" style="width: 250px;"/> </td>
      <td> <img src="Phase 2 media_output/img_1/ColorMap_1.png" alt="Color Map" style="width: 250px;"/> </td>
    </tr>
    <tr>
      <td align="center">Texton Map</td>
      <td align="center">Brightness Map</td>
      <td align="center">Color Map</td>
    </tr>
  </table>
</p>

#### Comparative Analysis
<p align="center">
  <table>
    <tr>
      <td> <img src="Images/canny_baseline_1.png" alt="Canny Edge Detection" style="width: 250px;"/> </td>
      <td> <img src="Images/sobel_1.png" alt="Sobel Edge Detection" style="width: 250px;"/> </td>
      <td> <img src="Phase 2 media_output/img_1/PbLite_1.png" alt="PBLite Detection" style="width: 250px;"/> </td>
    </tr>
    <tr>
      <td align="center">Canny Baseline</td>
      <td align="center">Sobel Baseline</td>
      <td align="center">PBLite (Our Method)</td>
    </tr>
  </table>
</p>

## Phase 2: Deep Dive on Deep Learning

### Overview
This phase implements and compares various convolutional neural network architectures for image classification on the CIFAR-10 dataset. The implementation includes several modern network architectures and optimization techniques to improve classification performance.

### Implemented Architectures
- **LeNet**: Classic convolutional neural network architecture
- **Custom CIFAR10 Model**: Tailored architecture with batch normalization
- **ResNet**: Deep residual network with skip connections
- **DenseNet**: Dense convolutional network with dense connectivity pattern
- **ResNeXt**: Advanced architecture with grouped convolutions and cardinality

### Key Features
- Batch normalization for faster training and better convergence
- Skip connections in ResNet and ResNeXt for deep network training
- Dense connectivity in DenseNet for better feature reuse
- Data augmentation including random crops and horizontal flips
- Learning rate optimization with Adam optimizer
- Comprehensive model evaluation and visualization

### Training
```bash
python Train.py [options]
```

#### Training Parameters
```
--CheckPointPath      Path to save checkpoints (default: ../Checkpoints/)
--NumEpochs          Number of training epochs (default: 25)
--DivTrain           Train data division factor (default: 1)
--MiniBatchSize      Training batch size (default: 128)
--LoadCheckPoint     Load from checkpoint? (0/1, default: 0)
--LogsPath           Training logs directory (default: LogsRes/)
```

### Model Architecture Details

#### CIFAR10 Custom Model
- 3 convolutional blocks with increasing channels (16→32→64→128→256)
- Batch normalization after each convolution
- MaxPooling layers for spatial dimension reduction
- Fully connected layers (512→128→10)

#### ResNet Implementation
- Residual blocks with skip connections
- Convolutional blocks with BatchNorm and ReLU
- Adaptive average pooling
- Dropout for regularization

#### DenseNet Architecture
- Dense blocks with growth rate of 12
- Transition layers for dimension reduction
- Global average pooling
- Dense connectivity pattern

#### ResNeXt Design
- Cardinality of 8 for grouped convolutions
- Three stages with increasing channels (128→256→512)
- Bottleneck blocks for efficient computation

### Model Evaluation
```bash
python Test.py [options]
```

#### Testing Parameters
```
--ModelPath          Path to trained model checkpoint
--LabelsPath         Path to test labels file
--SelectTestSet      Test set selection flag
--ModelType          Architecture selection
```

### Results Visualization
- Training and validation accuracy curves
- Loss progression plots
- Confusion matrix generation
- Performance metrics comparison

### Model Performance Analysis
The implementation includes:
- Real-time accuracy tracking
- Loss monitoring
- Model parameter counting
- Confusion matrix visualization
- Cross-architecture performance comparison

### Training Optimizations
- Data normalization (mean/std: (0.4914, 0.4822, 0.4465)/(0.2023, 0.1994, 0.2010))
- Data augmentation techniques:
  - Random cropping (32x32 with padding=4)
  - Random horizontal flips (p=0.5)
- Weight decay (1e-4) for regularization
- Adam optimizer with learning rate 1e-3

### Development Tools
- PyTorch for model implementation
- TensorBoard for training visualization
- Matplotlib for result plotting
- tqdm for progress tracking
- scikit-learn for metrics computation


## Project Structure
```
project-root/
├── BSDS500/            # Dataset directory
├── Checkpoints/        # Model checkpoints
├── Images/             # Result visualizations
├── Logs/              # Training/testing logs
├── Outputs/           # Generated outputs
├── Phase_1_media/     # Phase 1 visualizations
├── Phase 2 media_output/  # Phase 2 results
├── TxtFiles/          # Label and configuration files
├── Train.py           # Training implementation
├── Test.py            # Testing implementation
└── Wrapper.py         # Phase 1 implementation
```

## Citation
If you find this work useful in your research, please consider citing:
```bibtex
@article{WPI_CV_2025,
  title={Probabilistic Boundary Detection and CNN Performance Enhancement},
  author={[Your Name]},
  journal={RBE549 Course Project},
  institution={Worcester Polytechnic Institute},
  year={2025}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
