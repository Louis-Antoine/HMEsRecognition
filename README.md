# HMEsRecognition
This project was done in a team of two as part of Concordia's Pattern Recognition course in the Winter 2022 semester.
# Description
This is a simplified implementation of [this reasearch paper](https://arxiv.org/abs/1901.06763) <br>
The goal of the original paper is to improve a neural network that creates a LaTeX expression from an HME (handwritten mathematical expression). The paper deals with every kind of HME's, including two-dimensional HMEs, such as fractions, summations, etc. Due to the limited labeled data for such 2D HMEs, the authors of the papers applied distortions to the datasets that were available to see if the additional training data would improve the neural network (which it did).<br>
Our project only deals with basic mathematical symbols (digits, +, =, ...) that are written on a single line.

# Training

The neural network was trained on a subset (7680 images) of a [Handwritten math symbols dataset](https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols). While the dataset contains many more images, we needed a limited set of the data in order to replicate the results from the paper.

4 different distortions were then applied to each image, which grew the dataset to 46,080 images.

The neural network was built with PyTorch. It has an input layer of size 45^2 (one input node for each pixel of an image), two hidden layers of sizes 128 and 64, and an output layer of size 15.

# Process

To read an HME, the python application first loads the HME:

![step2](https://github.com/Louis-Antoine/HMEsRecognition/assets/60940273/84c1d691-72e7-4d4f-b6e0-229729aa7b50)

It then isolates each token and runs every token individually through the neural network to match it with a mathematical symbol.

![step3](https://github.com/Louis-Antoine/HMEsRecognition/assets/60940273/d6e1bf5f-24ad-4bab-ad98-08e5cd109397)

# Results

When compared with the same neural network that is only trained on the original 7680 images:
- Precision: 80.8% to 92.0%
- Recall: 78.9% to 91.8%
- F1-score: 78.7% to 91.8%
- Accuracy: 78.9% to 91.8%
