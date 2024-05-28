import os
import numpy as np 
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import matplotlib.pyplot as plt

# Load the dataset from the CSV file
email_df = pd.read_csv('data.csv')

# Drop columns that are not needed for anomaly detection
email_df = email_df.drop(columns=['id', 'to', 'cc', 'bcc', 'content'])

# Fill missing values with 0
email_df = email_df.fillna(0)

# Convert categorical features to numeric
email_df['date'] = pd.to_datetime(email_df['date'])
email_df['date'] = email_df['date'].astype(np.int64) // 10**9  # Convert to Unix timestamp
email_df['user'] = email_df['user'].astype('category').cat.codes
email_df['pc'] = email_df['pc'].astype('category').cat.codes
email_df['from'] = email_df['from'].astype('category').cat.codes

# Fit k-NN model on the dataset
nn_model = NearestNeighbors(n_neighbors=5)
nn_model.fit(email_df)

# Find distances and indices of k-neighbors for each sample
distances, indices = nn_model.kneighbors(email_df)

# Compute anomaly scores based on distances to neighbors
anomaly_scores = distances.mean(axis=1)

# Visualize the anomaly scores
plt.figure(figsize=(10, 6))
plt.hist(anomaly_scores, bins=50, alpha=0.7, color='blue')
plt.title('Anomaly Scores Distribution')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.show()

