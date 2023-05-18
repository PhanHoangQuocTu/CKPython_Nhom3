from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

# Use silhouette score to find optimal number of clusters to segment the data
num_clusters = np.arange(2,10)
results = {}
for size in num_clusters:
    model = KMeans(n_clusters = size).fit(data)
    predictions = model.predict(data)
    results[size] = silhouette_score(data, predictions)

best_size = max(results, key=results.get)