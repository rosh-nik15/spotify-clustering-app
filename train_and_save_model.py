# Save this in train_models.py
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN

df = pd.read_csv("spotifyFeatures.csv")
features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'valence', 'tempo']
X = df[features]
print("check1")
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)
print("check2")

kmeans = KMeans(n_clusters=7, random_state=42)
kmeans.fit(scaled_X)
print("check3")

from sklearn.decomposition import PCA

# Reduce to 3–5 dimensions
pca = PCA(n_components=5, random_state=42)
reduced_X = pca.fit_transform(scaled_X)

dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(scaled_X)
print("check4")

# Save models
joblib.dump(scaler, "scaler.pkl")
joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(dbscan, "dbscan_model.pkl")
print("check5")

