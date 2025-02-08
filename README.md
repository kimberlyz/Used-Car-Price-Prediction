# What Drives the Price of a Used Car?

## Overview

In this assignment, we were given the problem statement below.

Note: you can see the full code and more details in the Jupyter Notebook Prompt_II.ipynb.

**Problem Statement**

In this application, you will explore a dataset from Kaggle. The original dataset contained information on 3 million used cars. The provided dataset contains information on 426K cars to ensure the speed of processing. Your goal is to understand what factors make a car more or less expensive. As a result of your analysis, you should provide clear recommendations to your client—a used car dealership—as to what consumers value in a used car.

## Data Analysis

The data included the following features.
* id: unique identifier of car
* VIN: vehicle identification number
* region: ex - columbus, jacksonville, kansas city
* price: car sale price in USD
* year: car manufacture year
* manufacturer: ex - toyota, honda, jeep
* model: ex - f-150, camry, silverado
* condition: ex - good, excellent, like new, new
* cylinders: ex - 4 cylinder, 6 cylinder, 8 cylinder
* fuel: ex - gas, diesel, other
* odometer: odometer value (miles driven)
* title_status: ex - clean
* transmission: ex - automatic, manual
* drive: ex - 4wd, rwd, fwd
* size: ex - compact, mid-size, full-size
* type: ex - sedan, SUV, truck
* paint_color: ex - white, brown
* state: 2 letter state abbreviation

**Data Cleaning**

After analyzing the data, we ended up doing the following steps to clean the data
* Dropping categorical columns with two many unique values (if we one-hot-encoded them, the data dimensionality would increase too much and make it slow to train a model)
* Dropping columns with too many missing values
* Filling columns with the mode that had > 58% non-null values and < 95% non-null values
* Dropping rows with n/a values in columns with > 95% non-null values
* Removing outliers in the price and odometer 

**Data Preparation**
* One Hot Encode the categorical values
* Ordinal Encode the condition column

## Modeling
Tried a variety of different models and the best one ended up being the following

```
ordinal_encoder = OrdinalEncoder(categories = [['salvage', 'fair', 'good', 'excellent', 'like new', 'new']])

numeric_columns = X_train.select_dtypes(include=['number']).columns.tolist()
columns_to_one_hot_encode =

not_encoded_df.select_dtypes(include='string').columns.tolist()
columns_to_one_hot_encode.remove('condition')

column_transformer = make_column_transformer(
    (StandardScaler(), numeric_columns), 
    (OneHotEncoder(), columns_to_one_hot_encode),
    (ordinal_encoder, ['condition']),
    remainder='passthrough',
    force_int_remainder_cols=False
)

pipe = Pipeline([
    ('col_transformer', column_transformer),
    ('linreg', LinearRegression())
])
```

**Model Result Analysis**

By analyzing the RMSE, the R2 score, and the Linear Regression model coefficients, we were able to come to the following conclusions

* On average, we'll be $9,568.79 off when predicting a used car sale price.
* About 57% of the variation in price can be explained by the model features, so it's not a very accurate model, but can still yield some insights.
* Things the consumer care about in a used car that drive the sale price **up** are:
    * manufacturers - aston martin, tesla, porsche, datsun
    * model types - pickup, truck, rover
    * diesel fuel
    * title status - lien, clean
    * year (later year = newer car)
    * 10 cylinders, 12 cylinders, 8 cylinders
    * 4wd
* Things the consumer care about in a used car that drive the sale price **down** are:
   * odometer (high value for odometer = car is driven more and is more worn out)
   * fuel - electric
   * 3 cylinders
   * title_status - parts only
   * type - bus

## Takeaways / Recommendations

* When trying to buy used cars to re-sell, you can use the model to predict how much the used car will sell for, and then try to buy it at a cheaper price
* Continue collecting data for future sold cars, and re-train the ML model on the new input to see if you can get a better RMSE or R2 score (which will indicate a better model / more accurate price prediction)
   * When collecting data for future sold cars:
     * Think about collecting data (that you think will have a significant affect on price) outside the columns initially collected, and train a new ML model including those features
     * Think about collecting data from some columns that were dropped before (like size) and you might get enough data in the future to drop all the rows with n/a for size and still have a good amount of data to train on
