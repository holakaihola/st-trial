import io
import json
import pandas as pd
import numpy as np

import streamlit as st

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# st.markdown("<h4 style='color: #0066cc'>Estimate Future Price of Your Ideal Home</h4>", unsafe_allow_html=True)

st.markdown(
    """
    <div style='background-color: #0066cc; padding: 10px'>
        <h2 style='color: white;text-align: center;'>
        k-means clustering to predict CO2 emissions</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# st.markdown('''
# This is a dashboard showing the *forecasted prices* of different types of HDB :house: s
# ''')

### READ DATA ###
file_location = "USData.csv"
data = pd.read_csv(file_location, index_col=0)

data = data.rename(columns = {'Energy use (kg of oil equivalent per capita)':'Energy',\
                              'GDP per capita (current US$)':'GDP',\
                                  'CO2 emissions NET':'CO2'})

data = data.iloc[:56]    
data['Energy'] = data['Energy'].astype(float)
data['GDP'] = data['GDP'].astype(float)

# col1, mid, col2 = st.columns([2,0.5,2])

# Get the column names of the DataFrame
column_names = data.columns.tolist()

# Set the default selected columns
default_columns = ['Energy', 'GDP', 'CO2']

# Display the multi-select widget for selecting columns
st.markdown("<h4 style='color: #0066cc'>Select the variables to use in clustering</h4>", unsafe_allow_html=True)

selected_columns = st.multiselect("", column_names, default=default_columns)

# # Apply a theme to change the color of the multiselect widget
# st.markdown(
#     """
#     <style>
#     [data-baseweb="select"] { color: black !important; }
#     </style>
#     """,
#     unsafe_allow_html=True
# )


# Subset the data with the selected columns
selected_data = data[selected_columns]

# Cast the selected data columns as float
selected_data = selected_data.astype(float)

# Set the default number of clusters
default_clusters = 3

# Display the slider for selecting the number of clusters

st.markdown("<h4 style='color: #0066cc'>Select the Number of Clusters</h4>", unsafe_allow_html=True)

num_clusters = st.slider("", min_value=2, max_value=10, value=default_clusters)

# Extract the 3 numerical variables
X = data.iloc[:, 1:4].values

#Scale the data 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(selected_data)

# Perform K-means clustering
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(X_scaled)

# Get the cluster labels
labels_kmeans = kmeans.labels_
data['labels_kmeans'] = labels_kmeans

# Finding the final centroids
centroids = kmeans.cluster_centers_
# Evaluating the quality of clusters
s = metrics.silhouette_score(X_scaled, labels_kmeans, metric='euclidean')
print(f"Silhouette Coefficient for the Dataset Clusters: {s:.2f}")

# Plot the clusters
# , c=colors
fig, ax = plt.subplots()
ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_kmeans, cmap='viridis')
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', color='red', s=100)
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('K-means assigned Clustering on original df')

# Pass the figure object to st.pyplot()
st.pyplot(fig)
