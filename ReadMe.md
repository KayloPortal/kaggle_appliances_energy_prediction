# Appliances Energy Prediction

Quick access:
- [Objective](#objective)
- [How to run](#how-to-run)
- [Quick Look on The Dataset](#quick-look-on-the-dataset)
- [Comparing Models Performance](#model-performance-comparison)
  - [Models Performance Table](#models-performance-table)

## Objective

Given a dataset including houses, we're asked to train a model that can predict the amount of energy consumed by a house. I solve this problem using Regression and Regularization, while I explain the thought process, challenges, and strategies.

This readme file explains the theory behind every step I make, while you can find more information on how I implement the ideas by reading the notes in the notebooks.

- Dataset: [Appliances Energy Prediction Dataset from Kaggle](https://www.kaggle.com/datasets/loveall/appliances-energy-prediction)
- Libraries Used: Numpy, Matplotlib, Pandas

## How to Run

## Quick Look on The Dataset

Each row represents a house, for each we have humidty rate and temperature of different parts of the house and the outside environment. For every house, there are columns representing data gathered from a nearby weather station, which again tells us the humidity and temperature of the outside environment, with other information such as wind speed, pressure, etc. we're also provided with two random variables.

We're asked to estimate the amount of consumed energy by light fixtures in each house.

## First Approach
Because all values are continuous, a regression model could be used. We will have a lot of weights, which reduces model interpretiblity and can lead to overfitting. Hence, we use L1 or L2 regularization, to prevent overfitting and keep the model explainable while making accurate predictions. This will be our cost function:

$J_{λ}(w) = J(w) + λR(w)$

Higher λ results in less variance(More explainable model) but higher J(w) therefore higher bias. Lower λ results in higher variance(More complex model) but lower J(w) therefore lower bias. By optimizing the hyperparameter λ, we can find the sweetspot in bias-variance tradeoff for our model.

Exact value for λ and preferred norm(L1 or L2) will be determined in cross-validation phase. First we train two models, one with L1 and one with L2.

Here's the training timeline:
1. Split the data into training and test
2. Train two models on the same training set, ridge regularization and lasso regularization.
3. Find the best λ for each model.
4. Compare the models to choose the better one.

A proper regularization must determine which weights are more important and which are less important(features irrelevant to our target) so they end up with zero or close to zero value. Hence, rv1 and rv2 (random variables in the data set) should be nearly zero or excatly zero, in a good regularization, otherwise the model is hallucinating, seeing patterns and relations that doesn't exist.

Our data is noisy, it seems to have a lot of irrelevant features, however with a good regularization, we can reduce the effect of the noise on the model, and force the model to pay attention to the features that actually matter.

## Random Variables

As I mentioned, there are two columns of random variables that are irrelevant to our target variable(I'll use them later for model validation). Before training the model, Let's make sure they really have no relation with the target variable:

## Comparing Models Performance

### Models Performance Table