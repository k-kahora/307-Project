# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ISLP import load_data
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
AUTO = load_data("Auto")

# Question1
selected_coloumns = ['mpg','cylinders', 'displacement',  'weight', 'acceleration','horsepower' ]
X = AUTO[selected_coloumns]
y = AUTO['origin'] # Origin of car (1. American, 2. European, 3. Japanese)
print(X.head)
print("American cars: {0}".format((y == 1).sum())) # American
print("European cars: {0}".format((y == 2).sum())) # European
print("Japanese cars: {0}".format((y == 3).sum())) # Japanese

# Question2
print("Standard deviation")
print(X.std())
print("\n")
print("Mean")
print(X.mean())

# Question3_parta
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
components = pca.components_
x = np.arange(components.shape[1]) # 6
plt.plot(x, components[0], label='φ1')
plt.plot(x, components[1], label='φ2')
plt.xticks(ticks=x, labels=X.columns, rotation=45)
plt.xlabel('Features')
plt.ylabel('Principal Component Weights')
plt.title('Principal Component Weights per Feature')
plt.legend()
plt.tight_layout()
plt.savefig("plot.png", bbox_inches='tight')
plt.close()

# Question4
means = np.mean(X, axis=0)
stds = np.std(X, axis=0)
Z = (X - means) / stds
means_Z = np.mean(Z, axis=0)
stds_Z = np.std(Z, axis=0)
print(means_Z, stds_Z)

# Question5_parta
pca = PCA(n_components=2)
Z_reduced = pca.fit_transform(Z)
print("Reduced data size:", Z_reduced.shape)
components = pca.components_
plt.plot(x, components[0], label='φ1')
plt.plot(x, components[1], label='φ2')
plt.xticks(ticks=x, labels=Z.columns, rotation=45)
plt.xlabel('Features')
plt.ylabel('Principal Component Weights')
plt.title('Principal Component Weights per Feature')
plt.legend()
plt.tight_layout()
plt.savefig("plot-standard.png", bbox_inches='tight')
plt.close()
# plt.show()

# Question5_partc
dot_product = np.dot(components[0], components[1])
# Calculate the magnitude (norm) of each principal component to check if it's equal to one
magnitude_phi1 = np.linalg.norm(components[0])
magnitude_phi2 = np.linalg.norm(components[1])
print("Dot product", dot_product)
print("Magnitude phi 1",magnitude_phi1)
print("Magnitude phi 2", magnitude_phi2)

# Question6_parta
origins = ["American", "European", "Japanese"]
# plt.figure(figsize=(8, 6))
for origin in [1, 2, 3]:
    subset = Z_reduced[origin == y]
    plt.scatter(subset[:, 0], subset[:, 1], label=origins[origin - 1], alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Car Dataset by Origin')
plt.legend(title='Car Origin')
plt.tight_layout()
plt.savefig("plot-scatter.png")
plt.close()

# Assuming AUTO is a pandas DataFrame containing the dataset

# Calculate mean value of the feature (e.g., 'mpg') for each class
mean_mpg_american = AUTO[AUTO['origin'] == 1]['displacement'].mean()
mean_mpg_european = AUTO[AUTO['origin'] == 2]['displacement'].mean()
mean_mpg_japanese = AUTO[AUTO['origin'] == 3]['displacement'].mean()

# Display the mean values
print("Mean displacement for American cars:", mean_mpg_american)
print("Mean displacement for European cars:", mean_mpg_european)
print("Mean displacement for Japanese cars:", mean_mpg_japanese)

# Determine which class has the largest mean value for the feature
if mean_mpg_american > mean_mpg_european and mean_mpg_american > mean_mpg_japanese:
    print("American cars have the largest mean displacement.")
elif mean_mpg_european > mean_mpg_american and mean_mpg_european > mean_mpg_japanese:
    print("European cars have the largest mean displacement.")
else:
    print("Japanese cars have the largest mean displacement.")

mean_acceleration_american = AUTO[AUTO['origin'] == 1]['acceleration'].mean()
mean_acceleration_european = AUTO[AUTO['origin'] == 2]['acceleration'].mean()
mean_acceleration_japanese = AUTO[AUTO['origin'] == 3]['acceleration'].mean()

print("Mean acceleration for American cars:", mean_acceleration_american)
print("Mean acceleration for European cars:", mean_acceleration_european)
print("Mean acceleration for Japanese cars:", mean_acceleration_japanese)
print("European cars have the largest mean acceleration:", mean_acceleration_european)

pca = PCA(n_components=6)
Z_reduced = pca.fit_transform(Z)
print("here")
components = pca.components_
x = np.arange(components.shape[1]) # 6
print("here")

plt.plot(x, components[0], label='φ1')
plt.plot(x, components[1], label='φ2')
plt.plot(x, components[2], label='φ3')
plt.plot(x, components[3], label='φ4')
plt.plot(x, components[4], label='φ5')
plt.plot(x, components[5], label='φ6')
plt.xticks(ticks=x, labels=selected_coloumns, rotation=45)
plt.xlabel('Features')
plt.ylabel('Principal Component Weights')
plt.title('Principal Component Weights per Feature')
plt.legend()
plt.tight_layout()
plt.savefig("plot-pca-6.png", bbox_inches='tight')
plt.close()

# Question 7
pca = PCA(n_components=6)
pca.fit(Z)

# Extract explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# sets up and outputs the plot.
plt.plot(range(1,7), explained_variance_ratio, alpha=0.7, color='blue', label='Explained Variance',marker='o')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot of PCA Explained Variance')
plt.xticks(range(1, 7))
plt.legend()
plt.tight_layout()
plt.savefig("plot-scree.png", bbox_inches='tight')
plt.close()
