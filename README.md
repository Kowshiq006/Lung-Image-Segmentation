# UNet++ Segmentation Model for Respiratory Disease Detection

---

## Table of Contents
- [Introduction](#introduction)
  - [Motivation](#motivation)
  - [Objective](#objective)
- [Related Works](#related-works)
- [Methodology](#methodology)
- [Results](#results)
  - [Discussion](#discussion)
- [Conclusion](#conclusion)
- [Future Improvements](#future-improvements)
- [Dataset](#dataset)
- [References](#references)

---

## Introduction

### Motivation
Medical image segmentation is crucial for accurate and timely detection of respiratory diseases. Early diagnosis through lung imaging can significantly improve patient outcomes. With respiratory diseases being one of the leading causes of mortality worldwide, there is a pressing need for more accurate segmentation methods. 

UNet++ enhances the traditional U-Net model by reducing the semantic gap between the encoder and decoder, improving segmentation performance. This study explores the application of UNet++ for lung image segmentation to assist in early disease detection.

### Objective
The goal of this study is to implement UNet++ for segmenting lung images and evaluating its performance in comparison to traditional methods. The research aims to:
- Enhance segmentation accuracy using deep learning techniques.
- Investigate the feasibility of UNet++ in real-world clinical applications.
- Contribute to medical image analysis advancements for better diagnosis and treatment planning.

---

## Related Works
Lung image segmentation is a critical step in diagnosing lung diseases. Existing research has explored various architectures:
- **U-Net**: Demonstrated success in medical image segmentation.
- **Attention U-Net**: Incorporates attention mechanisms to focus on target structures.
- **H-DenseUNet**: Hybrid dense networks for better feature extraction.
- **UNet++**: Introduced nested and dense skip connections for improved segmentation accuracy.

This study builds on these approaches by implementing UNet++ for lung image segmentation.

---

## Methodology
The methodology followed these steps:
1. **Dataset Selection**: The **Lung Mask Image Dataset** from Kaggle was used.
2. **Preprocessing**:
   - Converted grayscale images to RGB.
   - Created training (3,500 images) and validation datasets (1,500 images).
3. **Model Implementation**:
   - Built a UNet++ model with convolutional layers, max-pooling, dropout (0.5), and skip connections.
   - Configured the model to match dataset requirements.
4. **Training**:
   - Optimizer: **Adam**
   - Loss Function: **Binary Crossentropy**
   - Training for **20 epochs** with **50 steps per epoch**.

---

## Results
### Accuracy and Loss
- The model achieved **83.5% accuracy** after **20 epochs**.
- Loss decreased to **29.3%**.

#### Training Performance
![Accuracy over Epochs](images/accuracy_plot.png)
![Loss over Epochs](images/loss_plot.png)

#### Validation Metrics
- **Validation Accuracy**: ~72%
- **Validation Loss**: ~50%-55%

---

## Discussion
The results indicate that UNet++ is highly adaptable to the U-Net framework, proving its flexibility and efficiency. Further improvements can be achieved by:
- Increasing training dataset size.
- Fine-tuning hyperparameters.
- Investigating model generalizability for different diseases.

---

## Future Improvements
- Fine-tuning hyperparameters to improve accuracy further.
- Expanding dataset for better generalization.
- Deploying as a web-based tool for real-world applications.

---

## Dataset
- **Lung Mask Image Dataset**: [Kaggle Link](https://www.kaggle.com/datasets/newra008/lung-mask-image-dataset)
- Extract the dataset into the `data/` directory.

## Conclusion
This study implemented UNet++ for lung image segmentation and demonstrated its potential in medical image analysis. The hierarchical design of UNet++ improved segmentation accuracy and boundary preservation. Future research can explore real-world deployment for clinical applications.

---

## References
1. Zhou, Z., Siddiquee, M. M. R., Tajbakhsh, N., & Liang, J. (2018). *UNet++: A nested U-Net architecture for medical image segmentation.* [Link](https://arxiv.org/abs/1807.10165)
2. WHO (2018). *The top 10 causes of death.* [Link](https://www.who.int/news-room/fact-sheets/detail/the-top-10-causes-of-death)
3. Naik, A., Edla, D.R. Lung Nodule Classification on Computed Tomography Images Using Deep Learning (2021).* [Link]([https://link.springer.com/article/10.1007/s11277-020-07762-1](https://doi.org/10.1007/s11277-020-07732-1))
4. Li, X., et al. (2018). *H-DenseUNet: Hybrid densely connected U-Net for medical image segmentation.* [Link](https://ieeexplore.ieee.org/document/8419855)
5. Ronneberger, O., et al. (2015). *U-Net: Convolutional networks for biomedical image segmentation.* [Link](https://arxiv.org/abs/1505.04597)

---

## Usage
### Training
```bash
python train.py --epochs 20 --batch_size 16 --lr 0.001
```

### Inference
```bash
python predict.py --input data/sample_image.png --output results/masked_image.png
```




