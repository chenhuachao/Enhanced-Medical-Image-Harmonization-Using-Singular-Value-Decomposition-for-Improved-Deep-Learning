# Enhanced Medical Image Harmonization Using Singular Value Decomposition for Improved Deep Learning Performance

This repository contains the implementation of a novel supervised image harmonization technique leveraging Singular Value Decomposition (SVD) to improve the performance and robustness of deep learning models in medical image processing. The approach is detailed in the paper "Enhanced Medical Image Harmonization Using Singular Value Decomposition for Improved Deep Learning Performance".


## Table of Contents

- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Usage](#usage)
- [Datasets](#Datasets)
- [License](#license)


## Project Overview

Medical image processing often suffers from discrepancies in image quality and brightness due to medical facilities and scanning equipment variations. These discrepancies can significantly hinder the training and generalization of deep learning models, especially when training and testing across different datasets. This project introduces an SVD-based image harmonization technique designed to address these issues. Our method ensures consistent image quality by preserving essential medical image features during normalization, thereby improving the robustness and accuracy of models trained on harmonized datasets across diverse healthcare environments.
![xx](fig3.pdf)


## Requirements

- Python 3.9
- NumPy
- OpenCV
- TensorFlow or PyTorch
- Matplotlib
- Install packages in requirements.txt


## Usage

Main Script (main.py)
This is the main entry point for running experiments and evaluations on the harmonized datasets.

Run the script with:

python main.py --dataset_name dataset_name --model_type model_type


## Datasets





## License
This project is licensed under the MIT License. See the LICENSE file for details.
