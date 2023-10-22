# Melbourne Housing Market Data Analysis and Prediction
![image](https://github.com/huseyincenik/machine_learning/assets/127469334/8be8e879-07f3-4d4a-a621-311388f3c0e9)


## Overview

This project analyzes the Melbourne Housing Market dataset to predict house prices using machine learning models. The dataset contains information about various attributes of houses in Melbourne, including suburb, number of rooms, property type, distance from the city center, and more.

## Dataset

The dataset used in this project can be found on Kaggle. You can access it by following this [link](https://www.kaggle.com/datasets/anthonypino/melbourne-housing-market/data).

## Project Structure

This project is divided into five main parts:

### Part 1: Data Cleaning
- The dataset was cleaned to handle missing values and outliers.

### Part 2: Handling with Missing Values
- Missing values in various columns were handled by different techniques, including median and mode imputation.

### Part 3: Handling with Outliers
- Outliers were detected and managed using the Z-score method.

### Part 4: Feature Engineering
- New features, such as building age and total room count, were created.

### Part 5: ML Prediction
- Different machine learning models, including linear regression, ridge, lasso, elastic net, and decision tree, were used to predict house prices.

## Model Selection

The analysis revealed that the data has non-linear characteristics, as evident from the low R-squared score in linear models. The improvement in other metrics, except for R-squared, after log transformation demonstrates significant enhancement in model performance.

Non-linear models, such as the Decision Tree model, showed increased metrics, further emphasizing the non-linear nature of the data. It is likely that better scores can be achieved by using models like Random Forest or XGBoost.

To mitigate multicollinearity effects, regularization methods like ridge and lasso were employed. Regularization helps to balance the trade-off between variance and bias and mitigate potential overfitting.

Due to the non-linear relationships in the data, superior results were obtained with tree-based models. The analysis included the application of various models, including linear, ridge, lasso, elastic net, and decision tree. Ultimately, the decision tree model yielded the best results.

## Results

The results demonstrate that decision tree regression is a suitable algorithm for predicting house prices in the Melbourne Housing Market dataset. Further improvements can be explored using other non-linear models.

## Links

- [Kaggle Notebook Link](https://www.kaggle.com/huseyincenik/melbourne-house-price-regression-exploration)
- [Github Notebook Link](https://github.com/huseyincenik/machine_learning/tree/main/Project/melbourne_house_price_regression_exploration)
- [Linkedin Account](https://www.linkedin.com/in/huseyincenik/)
