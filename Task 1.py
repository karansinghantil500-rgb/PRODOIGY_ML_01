# Importing libraries
import os
os.chdir(r"C:\AC DSA\aProjectML")   # ✅ set working directory

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ================================
# Step 1: Load Dataset
# ================================
data = pd.read_csv("train.csv")   # ✅ load dataset
print("Dataset Shape:", data.shape)

# Show first 10 rows (like in Jupyter)
print("\nPreview of Dataset:\n", data.head(10))  

# ================================
# Step 2: Select Important Features
# ================================
# We'll use: GrLivArea (sqft living area), BedroomAbvGr (bedrooms), FullBath (bathrooms)
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
X = data[features]
y = data['SalePrice']

# ================================
# Step 3: Train/Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================================
# Step 4: Train Linear Regression
# ================================
model = LinearRegression()
model.fit(X_train, y_train)

# ================================
# Step 5: Predictions
# ================================
y_pred = model.predict(X_test)

# ================================
# Step 6: Model Evaluation
# ================================
print("\nMean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# ================================
# Step 7: Visualization - Actual vs Predicted
# ================================
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Prices vs Predicted Prices")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.show()

# ================================
# Step 8: Distribution Plot
# ================================
plt.hist(y_test, bins=30, alpha=0.5, label="Actual Price")
plt.hist(y_pred, bins=30, alpha=0.5, label="Predicted Price")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.title("Distribution of Actual vs Predicted Prices")
plt.legend()
plt.show()

# ================================
# Step 9: Correlation Heatmap
# ================================
corr_matrix = data[['GrLivArea','BedroomAbvGr','FullBath','SalePrice']].corr()
plt.figure(figsize=(4, 2))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
