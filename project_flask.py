import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, render_template_string
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

app = Flask(__name__)

IMG_DIR = 'static/images'
os.makedirs(IMG_DIR, exist_ok=True)

# Home Page (Very minimal UI)
@app.route('/')
def home():
    return """
    <h1>Welcome to the KMeans Clustering and Regression App</h1>
    <p>Select an action:</p>
    <ul>
        <li><a href="/upload">Upload Dataset for Clustering</a></li>
        <li><a href="/predict">Upload Dataset and Predict Performance Index</a></li>
    </ul>
    """
    
# API Endpoint to perform K-Means Clustering
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            data = pd.read_csv(file)

            features = ['Avg_Credit_Limit', 'Total_Credit_Cards', 'Total_visits_bank', 
                        'Total_visits_online', 'Total_calls_made']
            data_selected = data[features]

            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_selected)

            inertia = []
            cluster_range = range(1, 11)
            for k in cluster_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(data_scaled)
                inertia.append(kmeans.inertia_)


            elbow_path = os.path.join(IMG_DIR, 'elbow_method.png')
            plt.figure(figsize=(8, 5))
            plt.plot(cluster_range, inertia, marker='o')
            plt.title('Elbow Method for Optimal Clusters')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Inertia')
            plt.savefig(elbow_path)
            plt.close()

            # After seeing the elbow plot, we determine that the optimal mnumber of clusters is 3
            optimal_clusters = 3
            kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
            kmeans.fit(data_scaled)
            data['Cluster'] = kmeans.labels_

            pca = PCA(n_components=2)
            data_pca = pca.fit_transform(data_scaled)

            cluster_path = os.path.join(IMG_DIR, 'cluster_visualization.png')
            plt.figure(figsize=(8, 5))
            for cluster in range(optimal_clusters):
                cluster_points = data_pca[data['Cluster'] == cluster]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')
            plt.title('KMeans Clustering Visualization')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.legend()
            plt.savefig(cluster_path)
            plt.close()

            return f"""
            <h1>Clustering Completed</h1>
            <h2>Elbow Method Graph</h2>
            <img src='/static/images/elbow_method.png' alt='Elbow Method'>
            <h2>Cluster Visualization</h2>
            <img src='/static/images/cluster_visualization.png' alt='Cluster Visualization'>
            <p><a href="/">Back to Home</a></p>
            """
    return """
    <h1>Upload Your Dataset for Clustering</h1>
    <form method='post' enctype='multipart/form-data'>
        <input type='file' name='file'>
        <input type='submit' value='Upload'>
    </form>
    """

# API Endpoint for Multiple Regression Model
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            data = pd.read_csv(file)

            # Mapping Categorical data to binary values
            if 'Extracurricular Activities' in data.columns:
                data['Extracurricular Activities'] = data['Extracurricular Activities'].map({'Yes': 1, 'No': 0})


            X = data[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']]
            y = data['Performance Index']
            model = LinearRegression()
            model.fit(X, y)

            # Had to use a small trick here by writing the model data in a binary file to persist the data 
            with open('model.pkl', 'wb') as f:
                pickle.dump((model, X.columns), f)

            # Input Form (Not the best frontend)
            return """
            <h1>Predict Performance Index</h1>
            <form method="POST" action="/result">
                <label for="Hours_Studied">Hours Studied:</label><br>
                <input type="number" name="Hours Studied" step="0.1" required><br><br>
                <label for="Previous_Scores">Previous Scores:</label><br>
                <input type="number" name="Previous Scores" required><br><br>
                <label for="Extracurricular_Activities">Extracurricular Activities (1 = Yes, 0 = No):</label><br>
                <input type="number" name="Extracurricular Activities" min="0" max="1" required><br><br>
                <label for="Sleep_Hours">Sleep Hours:</label><br>
                <input type="number" name="Sleep Hours" step="0.1" required><br><br>
                <label for="Sample_Question_Papers">Sample Question Papers Practiced:</label><br>
                <input type="number" name="Sample Question Papers Practiced" required><br><br>
                <button type="submit">Predict</button>
            </form>
            """
    return """
    <h1>Upload Dataset to Train Model</h1>
    <form method='post' enctype='multipart/form-data'>
        <input type='file' name='file'>
        <input type='submit' value='Upload'>
    </form>
    """

# API Endpoint to display the predicted score
@app.route('/result', methods=['POST'])
def result():
    with open('model.pkl', 'rb') as f:
        model, feature_names = pickle.load(f)

    input_data = {feature: [float(request.form[feature])] for feature in feature_names}
    user_data = pd.DataFrame(input_data)

    predicted_score = model.predict(user_data)[0]

    # Returns the result!
    return render_template_string("""
    <h1>Prediction Result</h1>
    <p><strong>Predicted Performance Index:</strong> {{ predicted_score }}</p>
    <p><a href="/">Back to Home</a></p>
    """, predicted_score=f"{predicted_score:.2f}")

if __name__ == '__main__':
    app.run(debug=True)

