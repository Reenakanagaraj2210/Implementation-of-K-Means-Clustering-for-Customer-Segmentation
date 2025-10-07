# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import dataset and print head,info of the dataset
2. check for null values
3. Import kmeans and fit it to the dataset
4. plot the graph using elbow method
5. print the predicted array
6. plot the customer segments

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Reena K
RegisterNumber:  212224040272
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data = pd.read_csv("Mall_Customers.csv")
print(data.info())
print(data.isnull().sum())
X = data.iloc[:, [3, 4]]
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()
km = KMeans(n_clusters=5, init="k-means++", random_state=42, n_init=10)
y_pred = km.fit_predict(X)
data["cluster"] = y_pred
plt.scatter(data[data["cluster"] == 0]["Annual Income (k$)"],
            data[data["cluster"] == 0]["Spending Score (1-100)"],
            color="red", label="Cluster 0")
plt.scatter(data[data["cluster"] == 1]["Annual Income (k$)"],
            data[data["cluster"] == 1]["Spending Score (1-100)"],
            color="blue", label="Cluster 1")
plt.scatter(data[data["cluster"] == 2]["Annual Income (k$)"],
            data[data["cluster"] == 2]["Spending Score (1-100)"],
            color="green", label="Cluster 2")
plt.scatter(data[data["cluster"] == 3]["Annual Income (k$)"],
            data[data["cluster"] == 3]["Spending Score (1-100)"],
            color="magenta", label="Cluster 3")
plt.scatter(data[data["cluster"] == 4]["Annual Income (k$)"],
            data[data["cluster"] == 4]["Spending Score (1-100)"],
            color="cyan", label="Cluster 4")
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
            s=300, color="yellow", marker="*", label="Centroids")
plt.title("Customer Segments")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()
```
## Output:

<img width="752" height="455" alt="Screenshot 2025-10-07 111204" src="https://github.com/user-attachments/assets/dd24c04d-b762-4ada-9794-1990a4eb79f7" />

<img width="598" height="453" alt="image" src="https://github.com/user-attachments/assets/f2c4c90c-ecd9-4dc7-aca7-261afd6946fa" />


<img width="573" height="453" alt="image" src="https://github.com/user-attachments/assets/c4d6b386-25b2-4cb9-aea6-6a0c3126ef4f" />

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
