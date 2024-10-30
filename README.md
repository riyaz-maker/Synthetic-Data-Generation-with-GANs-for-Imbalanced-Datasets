Synthetic Data Generation with GANs for Imbalanced Datasets

**Overview**
This project implements a Generative Adversarial Network (GAN) to generate synthetic data for imbalanced credit card fraud detection datasets. It helps address the common problem of class imbalance in fraud
detection by generating realistic synthetic fraud cases that maintain the statistical properties of real fraud transactions.

**Features**

Custom GAN architecture optimized for financial data
Dynamic batch size handling for efficient training
Automatic data preprocessing and scaling
Synthetic data generation with distribution matching
Visualization tools for comparing original and synthetic data
Progress monitoring and detailed logging
Memory-efficient batch processing for large datasets

**Requirements**

Copytensorflow >= 2.0.0
pandas
numpy
scikit-learn
matplotlib

**Model Architecture**
Generator

Input: Random noise vector (latent_dim)
Hidden layers: Dense layers with LeakyReLU activation and BatchNormalization
Output: Synthetic transaction features with tanh activation

Discriminator

Input: Transaction features
Hidden layers: Dense layers with LeakyReLU activation and Dropout
Output: Binary classification (real/fake) with sigmoid activation

**Data Format**
The input dataset should be a CSV file with the following characteristics:

Target variable column named 'Class' (0 for normal, 1 for fraud)
Optional 'Time' column (will be removed during preprocessing)
Numerical feature columns

**Output**
The model generates:

Synthetic fraud transactions matching the original data distribution
Distribution comparison plots
Training progress logs
