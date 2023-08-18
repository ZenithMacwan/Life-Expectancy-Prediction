# Life Expectancy Prediction

![Life Expectancy Prediction](https://img.shields.io/badge/Life%20Expectancy%20Prediction-Data%20Analysis%20Project-blue)

This repository contains a data analysis project focused on predicting life expectancy around the world using various predictor variables. The goal of this project is to gain insights into the factors that contribute to life expectancy and to build predictive models to estimate life expectancy accurately.

## Table of Contents

- [About](#about)
- [Dataset](#dataset)
- [Project Highlights](#project-highlights)
- [Steps Involved](#steps-involved)
- [Features](#features)
- [Getting Started](#getting-started)
- - [Installation](#installation)
- - [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Contributing](#contributing)
  
## About

This project aims to explore the relationship between various predictor variables and life expectancy across different countries. By employing linear regression, decision tree, and random forest algorithms, we seek to build accurate predictive models that can estimate life expectancy based on the provided features.

## Dataset

The dataset used for life expectancy prediction includes a diverse set of predictor variables and corresponding life expectancy values for countries across multiple years. Variables were cleaned and preprocessed to ensure data quality and suitability for analysis.

## Project Highlights

- Utilized linear regression, decision tree, and random forest algorithms for life expectancy prediction.
- Performed data cleaning and preprocessing to handle missing values and ensure data quality.
- Conducted feature selection and model evaluation using various metrics.
- Provided insights into the most influential predictors of life expectancy.

## Steps Involved

1. Data Cleaning: Converted categorical variables to factors and addressed missing values.
2. Linear Regression:
   - Dropped insignificant variables and outliers.
   - Addressed multicollinearity using VIF analysis.
   - Achieved improved model performance by iterative feature selection.
3. Decision Tree:
   - Utilized the `rpart` library for building a decision tree model.
   - Employed 5-fold cross-validation for model evaluation.
4. Model Evaluation:
   - Calculated metrics such as RMSE, MAPE, and MAE for each model.

## Features

- Data exploration and preprocessing.
- Linear regression analysis with iterative refinement.
- Decision tree modeling with visualization.
- Model evaluation using key metrics.

## Getting Started

To use and explore the life expectancy prediction models, follow these steps:

### Installation

Clone the repository:

```bash
git clone https://github.com/zenithmacwan/Life-Expectancy-Prediction.git
cd Life-Expectancy-Prediction
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Usage

Run the life expectancy prediction script:

```bash
python predict_life_expectancy.py
```

## Model Evaluation

The life expectancy prediction models are evaluated using key metrics such as RMSE (Root Mean Squared Error), MAPE (Mean Absolute Percentage Error), and MAE (Mean Absolute Error). These metrics provide insights into the accuracy and performance of the models in estimating life expectancy.

## Contributing

Contributions to this project are welcome. If you have suggestions, improvements, or would like to contribute to the project's development, feel free to open an issue or submit a pull request.

---

By Zenith Macwan(https://github.com/zenithmacwan)
