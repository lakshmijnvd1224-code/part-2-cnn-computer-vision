# Part 2 — Computer Vision Problem Formulation and CNN Prototype

## Overview
This project builds a CNN model to classify manufacturing product images
into 4 categories: normal, scratch, dent, and stain. It demonstrates
how deep learning can automate quality inspection in manufacturing.

## Dataset
- **Source:** https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing
- **Total Images:** 480 (120 per class)
- **Classes:** normal, scratch, dent, stain
- **Image Size:** 96x96 pixels, RGB
- **Note:** Dataset files are not uploaded to this repository. Please download from the source link above.

## Problem Type
Image Classification — each image is classified into one of 4 classes
representing different surface conditions of a manufactured product.

## Approach
1. Loaded and explored image dataset — 480 balanced images across 4 classes
2. Preprocessed images — resized to 64x64, normalized pixel values to 0-1
3. Built a CNN with 3 convolution blocks, flatten, dense, and dropout layers
4. Trained for 30 epochs using Adam optimizer and categorical crossentropy loss
5. Evaluated using confusion matrix and classification report
6. Generated sample predictions with confidence scores

## Results
| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Normal | 1.00 | 1.00 | 1.00 |
| Scratch | 0.96 | 1.00 | 0.98 |
| Dent | 1.00 | 0.96 | 0.98 |
| Stain | 1.00 | 1.00 | 1.00 |
| **Overall Accuracy** | | | **99%** |

## Observations
- The CNN achieved 99% accuracy on the test set with only 30 epochs
- All 4 classes were classified with very high precision and recall
- The balanced dataset helped the model learn all classes equally well
- Sample predictions show 100% confidence on most test images
- Mild overfitting observed in later epochs as training accuracy reached 100%

## CNN Concept Explanation

**What is Convolution?**
Convolution is like sliding a small filter (like a magnifying glass) across
the image to detect patterns such as edges, lines, and shapes. Each filter
learns to detect a specific feature automatically during training.

**Why is Pooling used?**
Pooling reduces the size of the feature maps while keeping the most important
information. Think of it like summarising a paragraph into one sentence —
it makes the model faster and less sensitive to small shifts in the image.

**Why is ReLU commonly used in CNNs?**
ReLU is simple and fast. It converts all negative values to zero and keeps
positive values as they are. This helps the network learn complex patterns
without slowing down training, unlike older activation functions like sigmoid.

**Why are CNNs better than regular networks for images?**
Regular feed-forward networks treat every pixel independently and do not
understand spatial relationships. CNNs use filters that look at groups of
nearby pixels together, allowing them to detect shapes, textures, and
patterns the way human eyes do. This makes CNNs far more efficient and
accurate for image data.

## Business Use Case — Manufacturing Quality Inspection

In a real manufacturing plant, cameras on the production line capture images
of every product as it moves along the conveyor belt. A CNN model like this
one can instantly classify each product as normal or defective, and identify
the type of defect (scratch, dent, or stain). This removes the need for
manual human inspection, reduces errors, speeds up the production process,
and saves significant cost. Defective products can be automatically rejected
before they reach customers, improving product quality and brand reputation.

## Repository Structure
part-2-cnn-computer-vision/
├── README.md
├── notebook.ipynb
├── requirements.txt
├── sample_predictions/
│   └── prediction_outputs.png
└── results/
    ├── accuracy_loss_curves.png
    └── confusion_matrix.png

## Libraries Used
- TensorFlow — building and training the CNN model
- scikit-learn — train-test split, evaluation metrics
- numpy — image array operations
- matplotlib — plotting accuracy/loss curves and predictions
- seaborn — confusion matrix heatmap
- Pillow — loading and resizing images    
