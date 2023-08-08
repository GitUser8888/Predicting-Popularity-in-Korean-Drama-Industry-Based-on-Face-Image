# Face Value: Using AI to Predict Stardom in South Korean Drama Industry

## Introduction

The global popularity of South Korean dramas and actors had seen a significant surge in the past decade, largely due to the increasing influence of the Korean Wave, also known as "Hallyu". The Korean Wave is a term used to describe the spread of South Korean culture around the world. The global reach of streaming platforms like Netflix has also significantly contributed to the international popularity of K-dramas and Korean actors.


## Motivation of this Project

The South Korean entertainment industry is fiercely competitive and burgeoning with potential talents. Current manual scouting methods may not fully capture the breadth of this potential
Scouting is also subject to the bias of the individual talent scouts themselves.

Other factors aside from looks determine how popular or successful the actor is, however, looks make the first impression, more so in the entertainment industry where visuals are the content. For other things like acting skills, dancing skills, vocals, these can be trained. On the other hand, for looks, we are either born with it or enhanced to a certain extent through invasive procedures.

This project aims to develop a predictive machine learning model that can aid the scouting process, where the potential popularity or success of the candidate or subject can be predicted based on their face image. The use case envisioned is a streamlined tool to automate analysis of potential where face images are systematically analysed and then flagged when the candidate or subject has a potential for popularity, where talent agencies can then recruit and further train the candidate or subject.

## Data Collection

Names, number of “hearts”, and face images of male South Korean actors from mydramalist.com are scraped in descending order of the number of "hearts". The number of "hearts" correspond to the popularity level of the actor, where the higher it is, the more popular the actor.

The data from a total of 490 South Korean male actors will be used in this modelling. The top 241 actors will be assigned the class "very popular", and the remaining 249 actors will be assigned "not so popular".

The face images are organised into separate folders with their respective popularity tier

## Modeling

The goal of this exercise is to find the best model, that is not overfitting, to classify face images between two classes, "very popular" and "not so popular".

A variety of pre-trained image classification models are used to extract features into numerical representations.

The pre-trained image classification models include:
- VGG16
- EfficientNetB0
- MobileNet
- VGGFace

PyCaret is used to systematically train different types of models with less code, and are then tuned and ensembled where necessary and where resources and time permits.

Certain hyperparameters like number of folds, train/test size, batch size, epochs etc. are tweaked for slightly different variations of the models to attempt to find the best model.

Autoencoders are also used to reduce dimensionality for higher speed of modelling and deployment.

Below is a summary table for the best models for each category: 

|Model Type|Model|Train Score|Test Score|AUC Score|
|:---:|:---:|:---:|:---:|:---:|
|<font color="blue">EfficientNetB0 + Autoencoder</font>|<font color="blue">Linear Discriminant Analysis</font>|<font color="blue">0.8513</font>|<font color="blue">0.8194</font>|<font color="blue">0.8853</font>|
|VGG16|Extreme Gradient Boosting|0.8571|0.7960|0.8622|
|VGG16 + Autoencoder|Extreme Gradient Boosting|0.7752|0.7520|0.8451|
|VGGFace|Extreme Gradient Boosting|0.8563|0.7724|0.8433|
|MobileNet + Autoencoder|Random Forest|0.8105|0.7725|0.8389|

---

The best model used EfficientNetB0 for feature extraction, and autoencoder for reduced dimensions, and Linear Discriminant Analysis, with a train accuracy of 0.8513 and test accuracy of 0.8194.

---

## Future Works

1. Expand definition of success to include more metrics including net worth, brand endorsements, drama ratings and others.

2. Expand regional scope to Japan, Taiwan, outside of Asia.

3. Add more actors, as well as many more images per actor.

4. Train using neural networks when there are more data, and explore adding a third class of popularity

5. Deploy model on a platform with more capabilities such as batch upload and dashboard
