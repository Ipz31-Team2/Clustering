from flask import Flask, jsonify, request
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

app = Flask(__name__)


# Load the wine dataset
wine = pd.read_csv('wine-clustering.csv')

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(wine)

# Perform PCA transformation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Perform k-means clustering
kmeans = KMeans(n_clusters=3)
kmeans_pca = kmeans.fit(X_pca)

@app.route('/wine/clustering', methods=['GET'])
def get_wine_clustering():
    y_pred = kmeans_pca.predict(X_pca)
    centroids = kmeans_pca.cluster_centers_  

    # Return the clusters and centroids
    result = {
        'clusters': y_pred.tolist(),
        'centroids': centroids.tolist()
    }
    return jsonify(result)


@app.route('/wine/predict', methods=['POST'])
def predict_wine_cluster():    
    # Extract the wine data from the request
    data = request.get_json()
    wine_data = np.array([[
        data['Alcohol'],
        data['Malic_Acid'],
        data['Ash'],
        data['Ash_Alcanity'],
        data['Magnesium'],
        data['Total_Phenols'],
        data['Flavanoids'],
        data['Nonflavanoid_Phenols'],
        data['Proanthocyanins'],
        data['Color_Intensity'],
        data['Hue'],
        data['OD280'],
        data['Proline']
    ]])  

    # Predict the cluster for the new wine data
    wine_data_pca = pca.transform(scaler.transform(wine_data))
    y_pred = kmeans.predict(wine_data_pca)

    # Return the predicted cluster
    result = {'cluster': int(y_pred[0])}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
