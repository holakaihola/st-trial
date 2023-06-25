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

st.markdown("<h4 style='color: #0066cc'>Estimate Future Price of Your Ideal Home</h4>", unsafe_allow_html=True)

st.markdown(
    """
    <div style='background-color: #0066cc; padding: 10px'>
        <h2 style='color: white;text-align: center;'>Forecasting Flat Prices</h2>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('''
This is a dashboard showing the *forecasted prices* of different types of HDB :house: s
''')


# TOWNS = ["ANG MO KIO", "BEDOK", "BISHAN", "BUKIT BATOK", "BUKIT MERAH", "BUKIT PANJANG", "BUKIT TIMAH", "CENTRAL AREA","CHOA CHU KANG", "CLEMENTI", "GEYLANG", "HOUGANG", "JURONG EAST", "JURONG WEST", "KALLANG/WHAMPOA", "MARINE PARADE", "PASIR RIS", "PUNGGOL", "QUEENSTOWN", "SEMBAWANG", "SENGKANG", "SERANGOON", "TAMPINES", "TOA PAYOH", "WOODLANDS", "YISHUN"]
# FLAT_TYPES = ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION']
# EXCLUSIONS = {
#     "ANG MO KIO, 1 ROOM", "BEDOK, 1 ROOM", "BISHAN, 1 ROOM", "BUKIT BATOK, 1 ROOM", "BUKIT PANJANG, 1 ROOM", "BUKIT TIMAH, 1 ROOM", "CENTRAL AREA, 1 ROOM", "CHOA CHU KANG, 1 ROOM", "CLEMENTI, 1 ROOM", "GEYLANG, 1 ROOM", "HOUGANG, 1 ROOM", "JURONG EAST, 1 ROOM", "JURONG WEST, 1 ROOM", "KALLANG/WHAMPOA, 1 ROOM", "MARINE PARADE, 1 ROOM", "PASIR RIS, 1 ROOM", "PUNGGOL, 1 ROOM", "QUEENSTOWN, 1 ROOM", "SEMBAWANG, 1 ROOM", "SENGKANG, 1 ROOM", "SERANGOON, 1 ROOM", "TAMPINES, 1 ROOM", "TOA PAYOH, 1 ROOM", "WOODLANDS, 1 ROOM", "YISHUN, 1 ROOM",
#     "BISHAN, 2 ROOM", "BUKIT BATOK, 2 ROOM", "BUKIT TIMAH, 2 ROOM", "MARINE PARADE, 2 ROOM",
#     "BUKIT MERAH, EXECUTIVE", "CENTRAL AREA, EXECUTIVE", "MARINE PARADE, EXECUTIVE",
#     "ANG MO KIO, MULTI-GENERATION", "BEDOK, MULTI-GENERATION", "BUKIT BATOK, MULTI-GENERATION", "BUKIT MERAH, MULTI-GENERATION", "BUKIT PANJANG, MULTI-GENERATION", "BUKIT TIMAH, MULTI-GENERATION", "CENTRAL AREA, MULTI-GENERATION","CHOA CHU KANG, MULTI-GENERATION", "CLEMENTI, MULTI-GENERATION", "GEYLANG, MULTI-GENERATION", "HOUGANG, MULTI-GENERATION", "JURONG EAST, MULTI-GENERATION", "JURONG WEST, MULTI-GENERATION", "KALLANG/WHAMPOA, MULTI-GENERATION", "MARINE PARADE, MULTI-GENERATION", "PASIR RIS, MULTI-GENERATION", "PUNGGOL, MULTI-GENERATION", "QUEENSTOWN, MULTI-GENERATION", "SEMBAWANG, MULTI-GENERATION", "SENGKANG, MULTI-GENERATION", "SERANGOON, MULTI-GENERATION", "TOA PAYOH, MULTI-GENERATION", "WOODLANDS, MULTI-GENERATION"
#     }

# town_flat_types = [t+', '+ft for ft in FLAT_TYPES for t in TOWNS if t+', '+ft not in EXCLUSIONS]
# defaults = town_flat_types[:2]

# options = st.multiselect(
#     'Select Town and Flat Type',
#     town_flat_types,
#     defaults,
#     max_selections=4)


# Set the default number of clusters
default_clusters = 3

# Display the slider for selecting the number of clusters
num_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=default_clusters)



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


# Extract the 3 numerical variables
X = data.iloc[:, 1:4].values

#Scale the data 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


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
