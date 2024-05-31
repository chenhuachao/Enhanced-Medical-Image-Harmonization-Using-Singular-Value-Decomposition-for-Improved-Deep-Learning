# Harmonizing-frequencies-across-medical-images-makes-learning-easier

This repository contains the implementation of a novel supervised image harmonization technique leveraging Singular Value Decomposition (SVD) to improve the performance and robustness of deep learning models in medical image processing. The approach is detailed in the paper "Enhanced Medical Image Harmonization Using Singular Value Decomposition for Improved Deep Learning Performance".

## Table of Contents

- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Code Description](#code-description)
- [License](#license)


## Project Overview

Medical image processing often suffers from discrepancies in image quality and brightness due to medical facilities and scanning equipment variations. These discrepancies can significantly hinder the training and generalization of deep learning models, especially when training and testing across different datasets. This project introduces an SVD-based image harmonization technique designed to address these issues. Our method ensures consistent image quality by preserving essential medical image features during normalization, thereby improving the robustness and accuracy of models trained on harmonized datasets across diverse healthcare environments.

## Requirements

- Python 3.9
- NumPy
- OpenCV
- TensorFlow or PyTorch
- Matplotlib
- Specifically in requirements.txt

## Installation

1. Clone the repository:
   git clone https://github.com/chenhuachao/Harmonizing-frequencies-across-medical-images-makes-learning-easier.git

   pip install -r requirements.txt

## Usage

Main Script (main.py)
This is the main entry point for running experiments and evaluations on the harmonized datasets.

Run the script with:
python main.py --dataset_name dataset_name --model_type model_type


## Code Description

har_fix.py

This script performs image harmonization using Singular Value Decomposition (SVD) on a batch of medical images.

Functions:

harmonize_image(image): Applies SVD to the input image and harmonizes it.

main_har_onechannel.py

Similar to har_fix.py but designed for single-channel images.

Functions:

harmonize_single_channel_image(image): Applies SVD to a single-channel image and harmonizes it.

main.py

This script is used to run different experiments and evaluate the harmonized datasets.

Experiment 1: MNIST and USPS Classification Experiment

Code in Handwritten Numbers Classification

1.Load Dataset Path

    base_dir = '/home/gem/Harry/C_To_SVD_USPS_use_USPS_0817_net/train'
    
2.Save model path

    checkpoints = '/home/gem/Harry/USPS_use_USPS_0818_net_model/checkpoints'

Experiment 1: MNIST and USPS Classification Experiment


## License
This project is licensed under the MIT License. See the LICENSE file for details.
