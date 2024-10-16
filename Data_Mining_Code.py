#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 17:10:47 2024

@author: PA
"""

import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Read the dataset
df = pd.read_csv("farming_job_duration.csv")

# Define functions for assigning numeric values to categorical variables
def assign_numeric_value_matterstate(row):
    if row['matterstate_Liquid'] == 1:
        return 1
    elif row['matterstate_Solid'] == 1:
        return 2
    else:
        return 0

def assign_numeric_value_producttype(row):
    if row['producttype_Seed'] == 1:
        return 1
    elif row['producttype_Rodenticide'] == 1:
        return 2
    elif row['producttype_Fertilizer']==1:
        return 3
    elif row['producttype_Herbicide']==1:
        return 4
    elif row['producttype_Insecticide']==1:
        return 5  
    else:
        return 0

def assign_numeric_value_fieldcrop(row):
    if row['fieldcrop_Alfalfa'] == 1:
        return 1
    elif row['fieldcrop_Barley'] == 1:
        return 2
    elif row['fieldcrop_Corn']==1:
        return 3
    elif row['fieldcrop_Cotton']==1:
        return 4
    elif row['fieldcrop_Grass']==1:
        return 5
    elif row['fieldcrop_Millet']==1:
        return 6
    elif row['fieldcrop_Oats']==1:
        return 7
    elif row['fieldcrop_Potatoes']==1:
        return 8
    elif row['fieldcrop_Rice']==1:
        return 8
    elif row['fieldcrop_Soybeans']==1:
        return 9
    elif row['fieldcrop_other']==1:
        return 10
    else:
        return 0

# Apply the functions to create new numeric columns
df['MatterState'] = df.apply(assign_numeric_value_matterstate, axis=1)
df['producttype'] = df.apply(assign_numeric_value_producttype, axis=1)
df['field_crop'] = df.apply(assign_numeric_value_fieldcrop, axis=1)

# Drop unnecessary columns
df = df.drop(df.columns[5:146], axis=1)

print(df)
# Split the data into features (X) and target variable (y)
X = df.drop('hoursdiff', axis=1)
y = df['hoursdiff']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_rmse = mean_squared_error(y_test, lr_pred, squared=False)
print("Linear Regression RMSE:", lr_rmse)

# Decision Trees
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_rmse = mean_squared_error(y_test, dt_pred, squared=False)
print("Decision Tree RMSE:", dt_rmse)

# Feature Importance for Decision Tree
feature_importances = dt.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)

# K-Nearest Neighbors
knn = KNeighborsRegressor()
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_rmse = mean_squared_error(y_test, knn_pred, squared=False)
print("KNN RMSE:", knn_rmse)

# Model selection using cross-validation
models = [lr, dt, knn]
model_names = ['Linear Regression', 'Decision Tree', 'KNN']
for model, name in zip(models, model_names):
    cv_rmse = np.sqrt(-cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error').mean())
    print(f"{name} Cross-validated RMSE:", cv_rmse)
    
# Regularization Techniques
# Ridge Regression
ridge = Ridge(alpha=1.0)  
ridge.fit(X_train, y_train)

ridge_rmse = np.sqrt(-cross_val_score(ridge, X, y, cv=5, scoring='neg_mean_squared_error').mean())
print("Ridge Regression RMSE:", ridge_rmse)

# Ridge Regression Model Coefficients
coefficients = ridge.coef_
feature_names = X.columns

coefficients_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
coefficients_df = coefficients_df.sort_values(by='Coefficient', ascending=False)
print(coefficients_df)

# Lasso Regression
lasso = Lasso(alpha=1.0) 
lasso.fit(X_train, y_train)

lasso_rmse = np.sqrt(-cross_val_score(lasso, X, y, cv=5, scoring='neg_mean_squared_error').mean())
print("Lasso Regression RMSE:", lasso_rmse)

# Model comparison with regularization
models = [ridge, lasso]
model_names = ['Ridge Regression', 'Lasso Regression']
cv_scores = []

for model, name in zip(models, model_names):
    cv_rmse = np.sqrt(-cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error').mean())
    cv_scores.append(cv_rmse)

    print(f"{name} Cross-validated RMSE:", cv_rmse)

best_model_idx = np.argmin(cv_scores)
best_model_name = model_names[best_model_idx]
best_model_rmse = cv_scores[best_model_idx]

print("\nBest Model:", best_model_name)
print("Best Model Cross-validated RMSE:", best_model_rmse)

from sklearn.preprocessing import PolynomialFeatures

# Generate interaction terms

poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_poly = poly.fit_transform(X_train[['orderid', 'totalacres', 'mnth']])
X_train_poly = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(['orderid', 'totalacres', 'mnth']))
X_train_extended = pd.concat([X_train, X_train_poly], axis=1)


from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Ridge tuning
ridge_params = {'alpha': [0.5, 0.75, 1.0, 1.25, 1.5]}
ridge_grid = GridSearchCV(Ridge(), ridge_params, cv=5, scoring='neg_mean_squared_error')
ridge_grid.fit(X_train, y_train)
print("Best Ridge RMSE:", np.sqrt(-ridge_grid.best_score_))

# KNN tuning
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsRegressor())
])
knn_params = {'knn__n_neighbors': range(1, 20)}
knn_grid = GridSearchCV(pipe, knn_params, cv=5, scoring='neg_mean_squared_error')
knn_grid.fit(X_train, y_train)
print("Best KNN RMSE:", np.sqrt(-knn_grid.best_score_))

from sklearn.model_selection import validation_curve

# Define range of alpha values for Ridge Regression
param_range = np.logspace(-2, 3, 10)
train_scores, test_scores = validation_curve(
    Ridge(), X_train, y_train, param_name="alpha", param_range=param_range,
    cv=5, scoring="neg_mean_squared_error", n_jobs=-1)

# Calculate mean and standard deviation for training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Calculate mean and standard deviation for test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

import matplotlib.pyplot as plt
model_names = ['Linear Regression', 'Decision Tree', 'KNN', 'Ridge Regression', 'Lasso Regression']
cv_rmse_values = [lr_rmse, dt_rmse, knn_rmse, ridge_rmse, lasso_rmse]

# Plotting the RMSE values
plt.figure(figsize=(10, 6))
plt.bar(model_names, cv_rmse_values, color=['blue', 'green', 'red', 'orange', 'purple'])
plt.xlabel('Model')
plt.ylabel('Cross-Validated RMSE')
plt.title('Comparison of Model Performance')
plt.ylim(0, max(cv_rmse_values) * 1.1)  # Set y-axis limit for better visualization
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
import numpy as np

# Assuming train_mean, train_std, test_mean, test_std, and param_range are already computed
plt.figure(figsize=(10, 6))
plt.plot(param_range, train_mean, 'b-', label='Training score', marker='o')  # Training score in blue solid line
plt.plot(param_range, test_mean, 'g--', label='Cross-validation score', marker='s')  # Validation score in green dashed line
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.1)
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color='green', alpha=0.1)

plt.title('Validation Curve for Ridge Regression')
plt.xlabel('Alpha')
plt.ylabel('Negative Mean Squared Error')
plt.xscale('log')
plt.legend(loc='best')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Define a broader range of alpha values
param_range = np.logspace(-6, 6, 13)

# Create a pipeline with scaling and Ridge Regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])

# Calculate validation curve
train_scores, test_scores = validation_curve(
    pipeline, X_train, y_train, param_name="ridge__alpha", param_range=param_range,
    cv=5, scoring="neg_mean_squared_error", n_jobs=-1
)

# Calculate mean and standard deviation for training and test sets
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot validation curves
plt.figure()
plt.semilogx(param_range, train_mean, label="Training score", color="darkorange", lw=2)
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="darkorange", alpha=0.2)
plt.semilogx(param_range, test_mean, label="Cross-validation score", color="navy", lw=2)
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="navy", alpha=0.2)

plt.title("Validation Curve with Ridge Regression (with Scaling)")
plt.xlabel("Alpha")
plt.ylabel("Negative Mean Squared Error")
plt.legend(loc="best")
plt.show()
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

# Create a pipeline with scaling and Ridge Regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])

# Define a range of alpha values to search through
param_grid = {'ridge__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}  # Adjust this range as needed

# Perform cross-validated grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best alpha value and corresponding RMSE
best_alpha = grid_search.best_params_['ridge__alpha']
best_rmse = np.sqrt(-grid_search.best_score_)
print("Best Alpha:", best_alpha)
print("Best RMSE:", best_rmse)
from sklearn.metrics import r2_score

# Fit the Ridge regression model
ridge = Ridge(alpha=1.0)  
ridge.fit(X_train, y_train)

# Make predictions
ridge_pred = ridge.predict(X_test)

# Calculate R-squared
r_squared = r2_score(y_test, ridge_pred)
print("R-squared for Ridge Regression:", r_squared)

# Plot predicted vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, ridge_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='red')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values (Ridge Regression)')
plt.show()
from sklearn.metrics import r2_score

# Assuming you have already trained your regression models and made predictions
# For example, let's consider the Ridge Regression model (ridge) and its predictions (ridge_pred)

# Calculate R-squared for Ridge Regression
r_squared_ridge = r2_score(y_test, ridge_pred)
print("R-squared for Ridge Regression:", r_squared_ridge)
