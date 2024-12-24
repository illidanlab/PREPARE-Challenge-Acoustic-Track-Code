# PREPARE-Challenge-Acoustic-Track-Code

Official Code of illidanlab team for PREPARE Challenge 2024 (Pioneering Research for Early Prediction of Alzheimer's and Related Dementias EUREKA Challenge 2024).

## Overview
Early detection of cognitive diseases is critical for effective treatment. While brain imaging has demonstrated high accuracy in disease detection, its widespread use is limited by complex technology and high costs, making it impractical for large-scale screening of adults. The PREPARE Challenge aims to develop effective methods leveraging acoustic biomarkers from patients' speech samples to identify cognitive characteristics. Our approach combines advanced acoustic feature extraction techniques, utilizing Wav2Vec and MFCC features, with robust outlier detection to mitigate noise. These features were then used in conjunction with Support Vector Machines (SVM), resulting in a competitive 11th-place ranking. Our codes are provided in [https://github.com/illidanlab/PREPARE-Challenge-Acoustic-Track-Code](https://github.com/illidanlab/PREPARE-Challenge-Acoustic-Track-Code). 

## Package dependencies
Use `conda env create -f environment.yml` to create a conda env and activate by `conda activate PREPARE`. 

## Demos
Here we provide several demos of results in the project report.
You can change the arguments from `main.py` to try different settings.
