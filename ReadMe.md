# Appliances Energy Prediction

Quick access:
- [Objective](#objective)
- [How to run](#how-to-run)
- [Quick Look on The Dataset](#quick-look-on-the-dataset)
- [Comparing Models Performance](#model-performance-comparison)
  - [Models Performance Table](#models-performance-table)

## Objective

Given a dataset, we're asked to predict appliances energy consumption by a house in a ten minute priod while we're given weather details, by using a proper regression model.

This readme file explains the theory behind every step I make, while you can find more information on how I implement the ideas by reading the notes in the notebooks.

- Dataset: [Appliances Energy Prediction Dataset from Kaggle](https://www.kaggle.com/datasets/loveall/appliances-energy-prediction)
- Libraries Used: Numpy, Matplotlib, Pandas, Scikit-learn, Seaborn

## How to Run

## Quick Look on The Dataset
Dataset includes information in 10 minute periods. In each row, humidty, temperature, and other quantities of the house are given for that ten minute period, plus information from a nearby weather station. Tempreture, humidity and other details of the outside evnironment is measured twice, one by the house sensors and one by the weather station.

We're also provided with two random variables.

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
5. Repeat the process above for polynomial regression and compare it to the current linear regression model

A proper regularization must determine which weights are more important and which are less important(features irrelevant to our target) so they end up with zero or close to zero value. Hence, rv1 and rv2 (random variables in the data set) should be nearly zero or excatly zero, in a good regularization, otherwise the model is hallucinating, seeing patterns and relations that doesn't exist.

Our data is noisy, it seems to have a lot of irrelevant features, however with a good regularization, we can reduce the effect of the noise on the model, and force the model to pay attention to the features that actually matter.

## Random Variables

As I mentioned, there are two columns of random variables that are irrelevant to our target variable(I'll use them later for model validation). Before training the model, Let's make sure they really have no relation with the target variable:

<img src="visuals\correlation_heatmap_rv1_appliance.png" alt="correlation heatmap between rv1 and target variable" style="float: left; margin-right: 20px; margin-bottom: 20px; max-height: 350px;">
<img src="visuals\correlation_heatmap_rv2_appliance.png" alt="correlation heatmap between rv2 and target variable" style="margin-bottom: 20px; max-height: 350px;">

As we can see in the correlation matrices, magnitudes of numbers assigned to correlation between target and v1, or target and v2, are very low ~0.01. We can conclude that there is no relation between random variables and the target, they're totally irrelavent to the target.

## Data Preprocessing

### Date column

The first column includes time in date format and is not clean continuous number, therefore its format needs to be changed.

Because times are ten minute values for 4-5 months, the absolute value of time doesn't matter here, There are periodic states like weekday or hour of the day, month, these are the metrics that actually have impact on the target variable. Hence, we replace this column with columns representing weekday and hour of the day.

If we extract one column for "hour", which contains numbers from 0 to 23, it will cause problems. Hour is a cyclic variable, the distance between every two consecutive hours is the same. But in our model, the number 23(11PM) is very far from the number 0(12AM), while in reality they're only one hour apart, just like other consecutive hours. If we use this (0, 23) numerical scale to represent hours, our model fails understanding that 11PM and 12AM are almost identical, it assumes they're very far, failing to understand the "Cyclical Continuity" that exists in our data.

To fix this, we have three good options:

- Using Sin/Cos to put hours on a circle (adds 2 columns)
- One-Hot encoding (adds 24 columns)
- Categorical encoding (adds 4 columns)

#### Using Sin/Cos

The first option adds two columns to represent hour:

1. `hour_sin`: $Sin(2\pi \cdot \text{hour} / 24)$
2. `hour_cos`: $Cos(2\pi \cdot \text{hour} / 24)$

To see what it does, imagine the unit circle $x^2 + y^2 = 1$ where for every dot on the circle $x=Cos(\alpha)$ and $y = Sin(\alpha)$. For every hour $\alpha = 2\pi \cdot \text{hour} / 24$ is an angle from $0$ to $2\pi$. Hence, these formulas put all our 24 hours on a unit circle(They assign a dot on the circle for each hour), in a way that all the dots are evenly spaced, creating the "Cyclical Continuity" we were looking for.

This way, the model thinks of every hour as a dot on the unit circle, where `hour_sin` and `hour_cos` show its exact coordinates. It tries to combine these triangular functions to create a wave-like function that shows the relation between 'hour of day' and 'energy consumed'.

This method doesn't add many columns hence it's efficient, but it only creates wave patterns, which wouldn't be able to catch some of the ups and downs and sudden jumps that could happen in a daily timeline. For instance, energy consumption might be constant from 3PM to 7PM but it may suddenly jump for thirty minutes, becuase someone has started using the oven for cooking, then the consumption would decrease to where it was. It's hard for a wave-like function constructed with Sin and Cos to catch such a jump, though not impossible.

#### One-Hot Encoding & Categorical Encoding

In One-Hot Encoding, we add 24 columns, each one for one hour of the day, if we're in that hour, its value will be one and other columns will be zero. The model learns to assign a weight for each hour seperately, showing the impact that single hour has on our target. This method is very effective in regression. Because it looks at each hour seperately, it can easily adupt to sudden jumps, and if given enough data, it's so flexible and can easily determine the impact of each hour on our target variable percisely. It's also so interpertable, after the model is learned we can see how energy consumption changes for every hour and find the peak hours.

However, it has a downside, it adds a lot of columns to our data, this is why sometimes people use Categorical Encoding(they add four columns for morning, afternoon, evening and night instead) or continue with Sin/Cos. However, because we have more than 20000 rows, even by adding 24 more columns we're in a safe zone for linear regression, what causes problem is polynomial regression.

| Method | One-Hot | Categorical | Sin/Cos |
| :--- | :--- | :--- | :--- |
| **Efficient** | Depends | Mostly | Mostly |
| **Flexible** | Very much | Good | Intermediate |
| **Cost** | Heavy | Low | Very low |

For every $x$ polynomial regression adds $x^2$, $x^3$, . . . depending on the degree and also the combination of terms such as $x \times y$ etc. (sckit-learn PolynomialFeatures) by adding columns for hours we have around 50 columns, in polynomial with degree two, we have 1,500 features and for degree three we have +25,000 features which is higher than the number of rows and leads us to overfitting.

But there's a solution to that. The 24 columns added from One-Hot encoding don't need to be powered, because $x^n = x$ for these columns. And because two hours never happen at the same time, $x \times y$ is always zero when both terms are from these columns. Hence they don't need to be multiplied by themselves. These reduce a lot of new features. We can be selective when creating new polynomial terms and reduce the number of features. I'm not sure if combination of hour columns with other columns is important or not, so I train a model with them and without them to observe the difference in performance.

We train different models using different options explained above with different methods to create polynomial terms, then we compare the performance to find the best model. Weekdays also need to be encoded with on of the three options above.

### Train, Test, Validation

We split the dataset into Train 80% and Test 20%. When training the model with regularization, K-Fold cross-validation is used to compare different values for $\lambda$, which splits training subset into K subsets, each one used once as the validation set.

Feature scaling is done using scikit-learn pipeline, this ensures scaling happens after each K-Fold subset creation, preventing data leakage. Each time the pipeline divides our training into a validation set and a new training subset, the scaling is only done on the new training subset so validation happens on validation set without data leakage.

## Comparing Models Performance

### Models Performance Table