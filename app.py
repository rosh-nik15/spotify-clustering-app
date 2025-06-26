import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import silhouette_score, davies_bouldin_score
import plotly.express as px

# Load models and data
kmeans = joblib.load("kmeans_model.pkl")
dbscan = joblib.load("dbscan_model.pkl")
scaler = joblib.load("scaler.pkl")
df = pd.read_csv("spotifyFeatures.csv")

# Assume features are pre-selected
features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'valence', 'tempo']
scaled_data = scaler.transform(df[features])

# Cluster Labels
df['KMeans_Cluster'] = kmeans.predict(scaled_data)
df['DBSCAN_Cluster'] = dbscan.fit_predict(scaled_data)
# Sidebar
st.sidebar.title("Spotify Recommender")
method = st.sidebar.selectbox("Select Clustering Algorithm", ["KMeans", "DBSCAN"])
selected_song = st.sidebar.selectbox("Select a Song", df['track_name'].unique())

# Title
st.title("🎵 Spotify Music Recommendation System")
st.markdown("Get recommendations based on song features using clustering.")

# Show cluster details
if method == "KMeans":
    cluster_label = df[df['track_name'] == selected_song]['KMeans_Cluster'].values[0]
    rec_songs = df[df['KMeans_Cluster'] == cluster_label].sample(5)
    st.subheader(f"🎧 Recommended Songs using KMeans")
else:
    cluster_label = df[df['track_name'] == selected_song]['DBSCAN_Cluster'].values[0]
    if cluster_label == -1:
        st.warning("This song is marked as an anomaly by DBSCAN. Try another.")
        rec_songs = pd.DataFrame()
    else:
        rec_songs = df[df['DBSCAN_Cluster'] == cluster_label].sample(5)
        st.subheader(f"🎧 Recommended Songs using DBSCAN")

# Display recommendations
if not rec_songs.empty:
    st.table(rec_songs[['track_name', 'artist_name', 'popularity']])

# Clustering Evaluation
st.subheader("📈 Clustering Evaluation Metrics")
silhouette_kmeans = silhouette_score(scaled_data, df['KMeans_Cluster'])
db_index_kmeans = davies_bouldin_score(scaled_data, df['KMeans_Cluster'])
st.markdown(f"**KMeans** - Silhouette Score: `{silhouette_kmeans:.2f}` | DB Index: `{db_index_kmeans:.2f}`")

if len(set(df['DBSCAN_Cluster'])) > 1:
    silhouette_dbscan = silhouette_score(scaled_data, df['DBSCAN_Cluster'])
    db_index_dbscan = davies_bouldin_score(scaled_data, df['DBSCAN_Cluster'])
    st.markdown(f"**DBSCAN** - Silhouette Score: `{silhouette_dbscan:.2f}` | DB Index: `{db_index_dbscan:.2f}`")

# Visualization
st.subheader("🧠 Cluster Visualization (2D PCA)")
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
df['PCA1'], df['PCA2'] = pca_data[:, 0], pca_data[:, 1]
fig = px.scatter(df, x='PCA1', y='PCA2', color='KMeans_Cluster' if method == "KMeans" else 'DBSCAN_Cluster',
                 hover_data=['track_name', 'artist_name'])
st.plotly_chart(fig, use_container_width=True)