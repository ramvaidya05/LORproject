KMeans Clustering and Regression Web App.

This application is designed to help you explore data by performing KMeans clustering and predict performance indices using a multiple linear regression model.

This app combines two major functionalities:

KMeans Clustering: Identify patterns or group similar data points in a dataset. It includes:
Elbow method visualization to find the optimal number of clusters.
2D PCA-based clustering visualization to observe how your data clusters.
Regression Prediction: Predict a "Performance Index" score based on features like study habits, extracurricular activities, and sleep hours. This feature is perfect for understanding how different factors contribute to outcomes.

Functionality: 
The app provides a simple interface with two main actions: Clustering and Prediction.

1. Kmeans Clustering
Upload a dataset containing credit and activity data. This can be found in the same directory as the .py file. It is called: "Clustering Credit Card Customer Data.csv"

2. Regression Prediction
Upload a dataset containing student exam performance. This can be found in the same directory as the .py file. It is called: "Regression_Student_Performance.csv"
Use the input form to predict the Performance Index by entering your own values for the above features.

3. Results
For Clustering:
View the Elbow Method and Cluster Visualization images.
For Regression:
See the predicted Performance Index based on your input.


How to Run?

Please run the below commands in your terminal:

pip install pandas numpy matplotlib flask scikit-learn
python project_flask.py

Open your browser and go to:
http://127.0.0.1:5000.

Packages Used:

Flask: Backend framework for serving web pages and handling user input.
Matplotlib: Visualization library for generating graphs.
Scikit-learn: For clustering, regression, and preprocessing.
Pandas: Data manipulation and analysis.
Pickle: Save and reload the trained regression model.
