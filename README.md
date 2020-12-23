# Pneumonia detection in X-ray images

Two CNN models were created to detect the presence of pneumonia in X-ray images and classify each image as "Normal" or "Pneumonia". Sourced from Kaggle dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

## Deep CNN model 

CNN model with 5 convolution blocks (using combinations of Conv2D, SeparableConv2D, BatchNormalization, MaxPool2D and Dropout) and a final Dense layer (single unit) with a sigmoid activation for binary classification. Model achieved 91.19% accuracy after training over 12 epochs with batch size of 32.  

## Transfer Learning model with Densenet201 architecture 

Customised Densenet201 Architecture with 3 added FC blocks (using Dense layer, Batch Normalization layer, Dropout layer) and a final Dense layer with a sigmoid activation for binary classification. Model achieved 94.39% accuracy after training over 12 epochs with batch size of 32. 

## Improvements & Future work

- Imbalanced dataset led to poor results on "Normal" images for both models. More images of "Normal" Pneomonia X-Rays are needed to improve accuracy. 
- K-fold validation for a better measure of the stochastic variance.
- Use of callbacks such as EarlyStopping, ReduceLROnPlateau and Modelcheckpoint improved the performance of Deep CNN model. Same techniques could also be applied to the Transfer Learning model to avoid overfitting. 

## References

This code is based on [madz2000's Kaggle notebook](https://www.kaggle.com/madz2000/pneumonia-detection-using-cnn-92-6-accuracy) and [Abhinav Sagar's Medium post](https://towardsdatascience.com/deep-learning-for-detecting-pneumonia-from-x-ray-images-fc9a3d9fdba8)
