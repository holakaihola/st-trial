import pandas as pd
import numpy as np

import streamlit as st

import matplotlib.pyplot as plt


from sklearn import metrics

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.patches import Patch


st.markdown(
    """
    <div style='background-color: #0066cc; padding: 10px'>
        <h2 style='color: white;text-align: center;'>
        k-means clustering to predict <br>CO2 emissions in 3 steps:</h2>
    </div>
    """,
    unsafe_allow_html=True
)

###########################################
######## Read Data ###############
###########################################
file_location = "USData.csv"
data = pd.read_csv(file_location, index_col=0)

data = data.rename(columns = {'Energy use (kg of oil equivalent per capita)':'Energy',\
                              'GDP per capita (current US$)':'GDP',\
                                  'CO2 emissions NET':'CO2 Emissions'})

data = data.iloc[1:56].astype(float)

# Get the column names of the DataFrame
column_names = data.columns.tolist()

# Set the default selected columns
default_columns = ['Energy', 'GDP', 'CO2 Emissions']

# Display the multi-select widget for selecting columns
st.markdown("<h4 style='color: #0066cc'>Step 1: Use the dropdown to select the variables to use in clustering</h4>", unsafe_allow_html=True)

selected_columns = st.multiselect("", column_names, default=default_columns)

# Validate the number of selected columns
if len(selected_columns) < 2:
    st.error("Please select at least two columns.")
    st.stop()
    
# Subset the data with the selected columns
selected_data = data[selected_columns]

# Cast the selected data columns as float
# selected_data = selected_data.astype(float)

# Set the default number of clusters
default_clusters = 3

# Display the slider for selecting the number of clusters

st.markdown("<h4 style='color: #0066cc'>Step 2: Move the slider below to select the Number of Clusters</h4>", unsafe_allow_html=True)

num_clusters = st.slider("", min_value=2, max_value=10, value=default_clusters)

###########################################
######## Scaling and Kmeans ###############
###########################################
st.markdown(f"<h4 style='color: #0066cc'>Step 3: Results: Evaluate Outputs for {num_clusters} clusters</h4>", unsafe_allow_html=True)

#Scale the data 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(selected_data)

# Perform K-means clustering
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(X_scaled)

# Add the cluster labels to the data
data['Cluster'] = kmeans.labels_

###########################################
######## PCA ##############################
###########################################
# Define a custom color mapping for up to 10 clusters
color_mapping = {
    0: 'tab:blue',
    1: 'tab:orange',
    2: 'tab:green',
    3: 'tab:red',
    4: 'tab:purple',
    5: 'tab:brown',
    6: 'tab:pink',
    7: 'tab:gray',
    8: 'tab:olive',
    9: 'tab:cyan'
}

# Perform dimensionality reduction using PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(selected_data)

# Add the cluster labels to the data
cluster_labels = kmeans.labels_
pca_data_with_clusters = pd.DataFrame(pca_data, columns=['Component 1', 'Component 2'])
pca_data_with_clusters['Cluster'] = cluster_labels

# Create a scatter plot of the data points with colors representing the clusters
plt.figure(figsize=(8, 6))
for cluster in range(num_clusters):
    cluster_data = pca_data_with_clusters[pca_data_with_clusters['Cluster'] == cluster]
    plt.scatter(cluster_data['Component 1'], cluster_data['Component 2'], label=f'Cluster {cluster + 1}')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('K-means Clustering PCA Visualization')
plt.legend()

# Display the plot using Streamlit
st.pyplot(plt)

###################################################
##### Performance metrics ####
###################################################
####### Section Title #######

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

silhouette_avg_kmeans = silhouette_score(X_scaled, cluster_labels)
calinski_score_kmeans = calinski_harabasz_score(X_scaled, cluster_labels)
davies_bouldin_kmeans = davies_bouldin_score(X_scaled, cluster_labels)

# Set Streamlit theme color
st.markdown(f'<style>body{{background-color: #0066cc;}}</style>', unsafe_allow_html=True)

# Display scores as a summary table
st.markdown(f"<h3 style='color: #0066cc;'>k-means {num_clusters} Cluster Performance Metrics</h3>", unsafe_allow_html=True)
st.subheader("**Silhouette Score:**")
st.markdown(f"<p style='color:#0066cc; font-size: 24px;'>{round(silhouette_avg_kmeans, 2)}</p>",unsafe_allow_html=True)

  # st.subheader("**Your monthly payment:**\n" + "$" + (monthly_payment, format='{:,d}'))
st.subheader("**Calinski-Harabasz Score:**")
st.markdown(f"<p style='color:#0066cc; font-size: 24px;'>{round(calinski_score_kmeans, 2)}</p>", unsafe_allow_html=True)

st.subheader("**Davies-Bouldin Score**")
st.markdown(f"<p style='color:#0066cc; font-size: 24px;'>{round(davies_bouldin_kmeans, 2)}</p>", unsafe_allow_html=True)
###################################################
##### Time Series plots for selected variables ####
###################################################
####### Section Title #######

st.markdown(
    f"""
    <div style='background-color: #5893d4; padding: 10px'>
        <h2 style='color: white;text-align: center;'>
        k-means {num_clusters} clusters overlayed on original Data</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# Define a dictionary mapping full column names to shortened versions
title_mapping = {
    'Energy': 'Energy',
    'GDP': 'GDP',
    'CO2 Emissions': 'CO2 Emissions',
    'Merchandise trade (% of GDP)': 'Merchandise trade',
    'Population density (people per sq. km of land area)': 'Population density',
    'Population growth (annual %)': 'Population growth',
    'Urban population growth (annual %)': 'Urban population growth',
    'Crop production index (2014-2016 = 100)': 'Crop production index',
    'Labor force participation rate, total (% of total population ages 15+) (national estimate)': 'Labor force participation rate',
    'Merchandise exports (current US$)': 'Merchandise exports',
    'Urban population': 'Urban population',
    'Cluster': 'Cluster'
}

fig, axs = plt.subplots(len(selected_columns), 1, figsize=(10, 10), sharex=True)

for i, column in enumerate(selected_columns):
    axs[i].plot(list(data.index), list(data[column]), color='gray')
    scatter = axs[i].scatter(data.index, data[column], c=data['Cluster'].map(color_mapping))
    axs[i].set_title(title_mapping[column])

# Get the unique cluster labels present in data['Cluster']
unique_clusters = data['Cluster'].unique()

# Create custom legend for the clusters present
legend_elements = []
for cluster in unique_clusters:
    legend_elements.append(Patch(facecolor=color_mapping[cluster], label=f'Cluster {cluster + 1}'))

plt.xlabel("Year")
# plt.suptitle('Selected Columns')
plt.tight_layout()

# Add the legend to the plot
plt.legend(handles=legend_elements)

st.pyplot(plt)

###################################################
##### Additional Charts ####
###################################################
# Create a list of columns not selected
st.markdown(f"<h3 style='color: #0066cc;'>Clusters overlayed on variables not used in clustering</h3>", unsafe_allow_html=True)

not_selected_columns = [column for column in column_names if column not in selected_columns]


fig, axs = plt.subplots(len(not_selected_columns), 1, figsize=(10, 10), sharex=True)

for i, column in enumerate(not_selected_columns):
    axs[i].plot(list(data.index), list(data[column]), color='gray')
    scatter = axs[i].scatter(data.index, data[column], c=data['Cluster'].map(color_mapping))
    axs[i].set_title(title_mapping[column])

# Get the unique cluster labels present in data['Cluster']
unique_clusters = data['Cluster'].unique()

# Create custom legend for the clusters present
legend_elements = []
for cluster in unique_clusters:
    legend_elements.append(Patch(facecolor=color_mapping[cluster], label=f'Cluster {cluster + 1}'))

plt.xlabel("Year")
# plt.suptitle('Selected Columns')
plt.tight_layout()

# Add the legend to the plot
plt.legend(handles=legend_elements)

st.pyplot(plt)
