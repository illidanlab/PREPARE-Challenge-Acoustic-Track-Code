# PREPARE-Challenge-Acoustic-Track-Code

Official Code of illidanlab team for PREPARE Challenge 2024 (Pioneering Research for Early Prediction of Alzheimer's and Related Dementias EUREKA Challenge 2024).

## Overview
![featured](pipeline.png)
Early detection of cognitive diseases is critical for effective treatment. While brain imaging has demonstrated high accuracy in disease detection, its widespread use is limited by complex technology and high costs, making it impractical for large-scale screening of adults. The PREPARE Challenge aims to develop effective methods leveraging acoustic biomarkers from patients' speech samples to identify cognitive characteristics. Our approach combines advanced acoustic feature extraction techniques, utilizing Wav2Vec and MFCC features, with robust outlier detection to mitigate noise. These features were then used in conjunction with Support Vector Machines (SVM), resulting in a competitive 10th-place ranking. Our codes are provided in [https://github.com/illidanlab/PREPARE-Challenge-Acoustic-Track-Code](https://github.com/illidanlab/PREPARE-Challenge-Acoustic-Track-Code). 

## Package dependencies
Use `pip install -r requirements.txt` to install necessary libraries. 

## Data Preparation
Download acoustic data from PREPARE Challenge website in [https://www.drivendata.org/competitions/299/competition-nih-alzheimers-acoustic-2/](https://www.drivendata.org/competitions/299/competition-nih-alzheimers-acoustic-2/) and put it in folder named `data`.

## Demos
Here we provide several demos of results in the project report.
You can change the arguments from `main.py` to try different settings.

- `--acoustic` (list of strings, optional, default: `["Wav2Vec", "MFCC"]`): 
  - Defines the acoustic features to use. 
  - Options include: `"Wav2Vec"`, `"MFCC"`.

- `--model` (string, optional, default: `"SVC"`): 
  - Specifies the classifier to use. 
  - Options include: `"MLP"`, `"SVC"`, `"GBC"`, `"RF"`, `"LR"`, `"NB"`.

- `--remove_noise` (flag, optional): 
  - Specifies type of noise to remove
  - Options include: `"all"`, `"short"`, `"language"`, `"alexa"`, `"svm"`, `"none"`.
  
- `--grid_search` (flag, optional): 
  - If set, perform GridSearch for the best hyperparameters of classifiers.
  - Use as `--grid_search` to enable.

### Examples

- Use MFCC as feature and Logistic Regression as classifier: `python main.py --acoustic MFCC --model LR`
- Remove noisy samples in training data: `python main.py --remove_noise`
- Use Wav2Vec + MFCC as features, Support Vector Classifier as classifier, Grid Search for hyper-parameters, and remove all noisy samples: `python main.py --acoustic Wav2Vec MFCC --model SVC --remove_noise all --grid_search`

