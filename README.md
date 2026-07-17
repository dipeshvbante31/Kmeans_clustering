# K-Means Clustering using Scikit-Learn

## Overview

This project demonstrates the implementation of the K-Means Clustering algorithm using Python and Scikit-Learn. It covers data generation, feature scaling, cluster formation, and techniques for determining the optimal number of clusters.

The notebook is designed for beginners who want to understand how unsupervised machine learning algorithms work and how clustering techniques are evaluated.

---

## Objectives

* Understand K-Means Clustering.
* Learn how to generate sample datasets.
* Perform feature scaling using StandardScaler.
* Apply the Elbow Method for selecting the optimal number of clusters.
* Validate clusters using KneeLocator and Silhouette Score.
* Visualize clustered data using Matplotlib.

---

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-Learn
* Kneed Library

---

## Project Workflow

1. Generate synthetic data using `make_blobs()`.
2. Visualize the generated dataset.
3. Split the dataset into training and testing sets.
4. Perform feature scaling using `StandardScaler`.
5. Apply K-Means clustering.
6. Find the optimal value of K using:

   * Elbow Method
   * Knee Locator
   * Silhouette Score
7. Predict cluster labels for test data.
8. Visualize the clustered output.

---

## Libraries Used

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from kneed import KneeLocator
```

---

## Dataset Information

The dataset is generated using:

```python
make_blobs(
    n_samples=1000,
    centers=3,
    n_features=2
)
```

### Parameters

| Parameter | Value |
| --------- | ----- |
| Samples   | 1000  |
| Clusters  | 3     |
| Features  | 2     |

---

## Feature Scaling

Feature scaling is performed using:

```python
StandardScaler()
```

Scaling ensures that all features contribute equally while calculating the distance between data points.

---

## K-Means Clustering

The K-Means algorithm groups similar data points into clusters based on their distances from the cluster centroids.

```python
KMeans(
    n_clusters=3,
    init="k-means++"
)
```

### Important Parameters

| Parameter     | Description                        |
| ------------- | ---------------------------------- |
| n_clusters    | Number of clusters                 |
| init          | Method for centroid initialization |
| fit()         | Trains the model                   |
| predict()     | Predicts cluster labels            |
| fit_predict() | Performs both operations together  |

---

## Finding the Optimal Number of Clusters

### 1. Elbow Method

The Within Cluster Sum of Squares (WCSS) values are calculated for multiple values of K.

```python
for k in range(1,11):
    model = KMeans(
        n_clusters=k,
        init="k-means++"
    )

    model.fit(x_train_scaled)

    wcss.append(model.inertia_)
```

The value where the graph forms an elbow is generally considered the optimal number of clusters.

---

### 2. Knee Locator

```python
from kneed import KneeLocator

knee = KneeLocator(
        range(1,11),
        wcss,
        curve="convex",
        direction="decreasing"
)

print(knee.elbow)
```

The Knee Locator automatically identifies the elbow point from the graph.

---

### 3. Silhouette Score

```python
from sklearn.metrics import silhouette_score
```

Silhouette Score measures how well-separated the clusters are.

```python
score = silhouette_score(
            x_train_scaled,
            model.labels_
        )
```

#### Interpretation

| Score      | Meaning              |
| ---------- | -------------------- |
| Close to 1 | Excellent clustering |
| Around 0   | Overlapping clusters |
| Negative   | Incorrect clustering |

Higher scores indicate better cluster separation.

---

## Visualizations

The notebook includes:

* Scatter plots
* Elbow curve visualization
* Cluster prediction visualization
* Silhouette score plots

These plots help understand cluster formation and model performance.

---

## Learning Outcomes

After completing this project, you will understand:

* Unsupervised Machine Learning
* K-Means Clustering
* Feature Scaling
* Cluster Validation Techniques
* Elbow Method
* Knee Locator
* Silhouette Score
* Data Visualization using Matplotlib

---

## Installation

Install the required libraries:

```bash
pip install numpy pandas matplotlib scikit-learn kneed
```

---

## How to Run

1. Clone the repository.
2. Install the required libraries.
3. Open the Jupyter Notebook.
4. Run all the cells sequentially.
5. Observe the visualizations and clustering results.

---

## Future Improvements

You can further improve this project by:

* Using real-world datasets.
* Performing Principal Component Analysis (PCA).
* Comparing K-Means with:

  * DBSCAN
  * Hierarchical Clustering
  * Gaussian Mixture Models
* Building an interactive clustering dashboard using Streamlit.

---

## Author

**Dipesh Vishnu Bante**

* B.Tech (CSE) Student
* Aspiring Data Scientist
* Skilled in Python, SQL, Machine Learning, Feature Engineering, and Data Analysis.
