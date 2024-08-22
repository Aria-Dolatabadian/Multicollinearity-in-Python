import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Step 1: Read Data from CSV
df = pd.read_csv('correlated_data.csv')
# Step 2: Multicollinearity Analysis - Calculate VIF (Variance Inflation Factor)
# VIF Calculation
vif_data = pd.DataFrame()
vif_data["feature"] = df.columns
vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
# Display VIF
print(vif_data)
# Step 3: Visualize Correlations Using a Heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()
# Step 4: Visualize Pairwise Relationships Using Scatter Plots
sns.pairplot(df)
plt.suptitle('Scatter Plot Matrix of Correlated Variables', y=1.02)
plt.show()

