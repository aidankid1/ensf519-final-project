# ENSF 519 - Final Project
*Due: 12/05/2025*

| Student | UCID |
| ----------- | ----------- |
| Aidan Huang | 30149948 |
| Jin Kim | 30173509 |
| Rohan Lange |  ... |

## Problem We Are Trying To Solve
Welcome to our ENSF 519 Final Project! We have been tasked by Meta (not actually) to develop a prototype ML model for facial emotion classification. This is in an effort to improve Meta's Instagram/Facebook selfie filter performance.

## Dataset Description
The dataset used for training can be accessed through:
[FER-2013 Kaggle Dataset](https://www.kaggle.com/datasets/msambare/fer2013)

This dataset was chosen because:

1. 700+ community posts using and testing this dataset with results
2. 48x48 pixel images allow for quick downloads
3. Comes pre-split into training and testing splits

## What Models We Used To Solve The Problem
1. Custom CNN
    - Designed for 48x48 images
    - Consist of ?? convolutional blocks (Conv -> ReLU -> Maxpool)
    - ?? fully connected layers
    - Regularization (dropout, data augmentation)
    - Full control over hyperparams

Foundational model as a baseline to understand performance.

2. Pre-trained ResNet18
    - 

## Our Results
write here. include ss's here.

## Interpretation Of Our Results
write here.

## Any Deviations From Proposal?
The only deviation from our proposal was excluding a dedicated front-end layer to maximize simplicity and reduce our scope during development. All other aspects of our project closely match our written proposal.

## Ethical Concerns?
**Ethical Concern 1 - Privacy**

Facial emotion recognition inherently involves analyzing sensitive biometric data.

*Potential issues include:*
 - Unauthorized coll ection or storage of facial images
 - Misuse in surveillance systems without user consent
 - Risk of re-identification if model outputs are linked back to individuals

*Mitigation strategies:*
 - Ensure all training data is publicly available and anonymized
 - Do not store user images locally or on servers
 - Avoid linking predictions to identifiable personal information

**Ethical Concern 2 - Bias**

Emotion datasets often contain uneven representation across demographics such as age, ethnicity, or gender.

*Potential:*
 - Unequal model performance across population subgroups
 - Stereotyping or misclassification of certain demographics
 - Potential harm if deployed in sensitive contexts (e.g., hiring, policing)

*Mitigation strategies:*
 - Acknowledge dataset limitations (e.g., FER-2013’s demographic imbalance)
 - Avoid deploying the model in real-world decision-making
 - Evaluate model performance across different categories where possible

**Ethical Concern 3 - Environmental**

Training deep learning models consumes computational resources and electricity, contributing to CO₂ emissions.
Although our models are relatively lightweight, large-scale facial analysis is environmentally costly.

*Mitigation strategies:*
 - Use transfer learning (as we did) to reduce compute time
 - Avoid unnecessary retraining
 - Run training on efficient hardware (e.g., GPUs) to minimize wasted energy

## How To Run
**Step 1 - Initialization**

Ensure your computer or environment has the following:
```
python --version: 3.10 or higher
```

>*Note: Model training time is significantly shorter if your device is using a supported **Nvidia GPU**.*


**Step 2 - Running Application**

Navigate to the `/src` directory:
```
cd src
```

Run the `moodify.py` application:
```
python moodify.py
```

**Step 3 - Moodify**

You will be able to select the model, re-train the models, and run an inference with your own desired images.