import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Install missing libraries if needed
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ModuleNotFoundError as e:
    import os
    os.system('pip install matplotlib seaborn')

# Load dataset (replace with Kaggle dataset link or path)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/StudentsPerformance.csv"
df = pd.read_csv(url)

# Data Preprocessing
# Rename columns for easy access
df.columns = df.columns.str.replace(" ", "_").str.lower()

# Encode categorical data
df = pd.get_dummies(df, drop_first=True)

# Features and Target
X = df.drop("math_score", axis=1)  # Predicting math score as example
y = df["math_score"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lr = lin_reg.predict(X_test)

# Decision Tree Regressor
dtree = DecisionTreeRegressor(random_state=42)
dtree.fit(X_train, y_train)
y_pred_dt = dtree.predict(X_test)

# Model Evaluation
def evaluate_model(y_test, y_pred, model_name):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}\n")

evaluate_model(y_test, y_pred_lr, "Linear Regression")
evaluate_model(y_test, y_pred_dt, "Decision Tree")

# Hyperparameter Tuning for Decision Tree
param_grid = {'max_depth': [3, 5, 10, None], 'min_samples_split': [2, 5, 10]}
grid_search = GridSearchCV(dtree, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

print("Best Parameters for Decision Tree:", grid_search.best_params_)

# Visualization
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred_lr, color='blue', label='Linear Regression')
sns.scatterplot(x=y_test, y=y_pred_dt, color='green', label='Decision Tree')
plt.plot([0, 100], [0, 100], 'r--')
plt.xlabel('Actual Scores')
plt.ylabel('Predicted Scores')
plt.title('Actual vs Predicted Student Scores')
plt.legend()
plt.show()
