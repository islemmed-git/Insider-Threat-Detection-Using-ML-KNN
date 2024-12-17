# Insider-Threat-Detection-Using-ML-KNN

# Email Anomaly Detection

## Overview
This project performs anomaly detection on email metadata using a **K-Nearest Neighbors (KNN)** algorithm. The goal is to detect unusual email activities based on features like the email size, sender, recipient, attachments, and timestamps. 

The anomalies are scored based on their distances to the nearest neighbors, and their distribution is visualized using a histogram.

---

## Requirements
Make sure you have the following libraries installed:
- **Python 3.x**
- **NumPy**
- **Pandas**
- **scikit-learn** (for K-Nearest Neighbors)
- **Matplotlib**

To install the required libraries, use the following command:
```bash
pip install numpy pandas scikit-learn matplotlib
```

---

## Dataset
The dataset `data.csv` contains the following email metadata:

| Column       | Description                                         |
|--------------|-----------------------------------------------------|
| `id`         | Unique identifier for each email                    |
| `date`       | Date and time the email was sent                   |
| `user`       | User who sent the email                            |
| `pc`         | Computer or machine identifier                     |
| `to`         | Recipients of the email                            |
| `cc`         | Carbon copy recipients                             |
| `bcc`        | Blind carbon copy recipients                       |
| `from`       | Sender of the email                                |
| `size`       | Size of the email in bytes                         |
| `attachments`| Number of attachments in the email                 |
| `content`    | Email body content                                 |


---

## Code Explanation
### 1. Import Libraries
The required libraries such as `numpy`, `pandas`, and `NearestNeighbors` from `scikit-learn` are imported. `matplotlib` is used for visualization.
```python
import os
import numpy as np 
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
```

### 2. Load the Dataset
The dataset is read from `data.csv` and unnecessary columns are removed.
```python
email_df = pd.read_csv('data.csv')
email_df = email_df.drop(columns=['id', 'to', 'cc', 'bcc', 'content'])
```

### 3. Handle Missing Data
Missing values are filled with `0`.
```python
email_df = email_df.fillna(0)
```

### 4. Convert Features to Numeric
To process categorical variables such as `user`, `pc`, and `from`, these are converted into numerical codes. The `date` column is converted to Unix timestamps.
```python
email_df['date'] = pd.to_datetime(email_df['date'])
email_df['date'] = email_df['date'].astype(np.int64) // 10**9  # Convert to Unix timestamp
email_df['user'] = email_df['user'].astype('category').cat.codes
email_df['pc'] = email_df['pc'].astype('category').cat.codes
email_df['from'] = email_df['from'].astype('category').cat.codes
```

### 5. K-Nearest Neighbors Model
The `NearestNeighbors` model is used to calculate the distances to the 5 nearest neighbors for each data point.
```python
nn_model = NearestNeighbors(n_neighbors=5)
nn_model.fit(email_df)
```

### 6. Calculate Anomaly Scores
The anomaly scores are the mean distances to the nearest neighbors.
```python
distances, indices = nn_model.kneighbors(email_df)
anomaly_scores = distances.mean(axis=1)
```

### 7. Visualize Anomaly Scores
A histogram of the anomaly scores is plotted to observe the distribution.
```python
plt.figure(figsize=(10, 6))
plt.hist(anomaly_scores, bins=50, alpha=0.7, color='blue')
plt.title('Anomaly Scores Distribution')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.show()
```

---

## Running the Code
1. Ensure `data.csv` is in the same directory as the script.
2. Run the script using Python:
   ```bash
   python anomaly_detection.py
   ```
3. The script will display a histogram showing the distribution of anomaly scores.

---

## Output
- A histogram plot that shows the distribution of anomaly scores. Higher scores correspond to more anomalous data points (emails).

---

## Customization
You can customize the number of neighbors in the KNN model:
```python
nn_model = NearestNeighbors(n_neighbors=5)  # Change 5 to your desired value
```

To explore anomalies further, you can identify emails with the highest anomaly scores:
```python
email_df['anomaly_score'] = anomaly_scores
print(email_df.sort_values(by='anomaly_score', ascending=False).head(10))
```
This will display the top 10 most anomalous emails.

---

## Use Case
This project can be extended to detect anomalies in any structured tabular data, such as:
- Network traffic logs
- Transaction records
- User activity logs

---

## Author
Your Name

## License
MIT License
