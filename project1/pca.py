# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ISLP import load_data
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load Iris dataset

AUTO = load_data("Auto")
selected_coloumns = ['mpg','cylinders', 'displacement',  'weight', 'acceleration', ]
X = AUTO[selected_coloumns]
y = AUTO['origin'] # Origin of car (1. American, 2. European, 3. Japanese)

#Standardize the data to have mean 0 and variance 1
scaler = StandardScaler()
scaler.fit(X)
X_standardized = scaler.transform(X)

print("Means: ",np.mean(X_standardized,axis=0))
print("Standard deviations: ",np.std(X_standardized,axis=0))

print("Original data size:",X.shape)
# Implement PCA with 2 principal components
pca = PCA(n_components=2)

# Fit the PCA model to the data and transform the data
X_reduced = pca.fit_transform(X_standardized)
print("Reduced data size:", X_reduced.shape)

#principal components
components = pca.components_
x = np.arange(components.shape[1]) # 6

# Plot the first and second principal components
plt.plot(x, components[0], label='φ1')
plt.plot(x, components[1], label='φ2')

# Define feature names as tick labels
names = ['mpg', 'cylinders', 'displacement', 'weight', 'acceleration']
plt.xticks(ticks=x, labels=selected_coloumns, rotation=45)

# Labeling the axes and the legend
plt.xlabel('Features')
plt.ylabel('Principal Component Weights')
plt.title('Principal Component Weights per Feature')
plt.legend()

# Show plot
plt.tight_layout()
plt.show()
