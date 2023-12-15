#!/usr/bin/env python
# coding: utf-8

# In[1]:


#GMM


# In[2]:


# Data Pre-Proccessing


# In[3]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import time
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn import mixture, cluster
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import SpectralClustering


# In[4]:


ds = pd.read_excel('Student_ds_100_n.xlsx')


# In[5]:


ds.head()


# In[6]:


ds.shape 


# In[7]:


ds.describe()


# In[8]:


ds.dtypes 


# In[9]:


ds.info() 


# In[10]:


ds.isnull().sum()


# In[11]:


plt.figure(1, figsize=(12,4))
n = 0
for x in ['SSLC', 'HSLC', 'Cgpa']:
    n+=1
    plt.subplot(1,3,n)
    plt.subplots_adjust(hspace= 0.5, wspace=0.5)
    sns.distplot(ds[x], bins = 20)
    plt.title('Distplot of {}'.format(x))
plt.show()


# In[12]:


# Training the algorithm and Visualization


# In[13]:


X = ds[['SSLC', 'HSLC', 'Cgpa']]


# In[14]:


gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)


# In[15]:


labels = gmm.predict(X)


# In[16]:


ds['Cluster Label'] = labels


# In[17]:


sns.relplot(x="HSLC", y="SSLC", hue="Cluster Label", size="Cgpa", sizes=(50, 200), data=ds)


# In[18]:


# Clusters into excel


# In[19]:


ds['Cluster'] = labels
high_performing_students = ds[ds['Cluster'] == 2]
medium_performing_students = ds[ds['Cluster'] == 1]
low_performing_students = ds[ds['Cluster'] == 0]


# In[20]:


writer = pd.ExcelWriter('Segmented_Model_GMM_100_ds.xlsx')
high_performing_students.to_excel(writer, sheet_name='High Performing Students')
medium_performing_students.to_excel(writer, sheet_name='Medium Performing Students')
low_performing_students.to_excel(writer, sheet_name='Low Performing Students')
writer.save()


# In[21]:


# Calculating Metrics for Comparision GMM


# In[22]:


X_norm = (X - X.mean()) / X.std() #Normalizing the data


# In[23]:


gmm = mixture.GaussianMixture(n_components=3)
gmm.fit(X)
labels = gmm.predict(X)
ds['Cluster'] = labels


# In[24]:


start_time = time.time()
gmm.fit(X_norm)
gmm_labels = gmm.predict(X_norm)
gmm_time = time.time() - start_time


# In[25]:


# Calculating Accuracy
gmm_accuracy = np.mean(gmm_labels == ds['Cluster'])


# In[26]:


#Accuracy with noise
noise = np.random.normal(0, 0.1, size=X_norm.shape)
X_noisy = X_norm + noise
gmm_labels = gmm.predict(X_noisy)
gmm_accuracy_nis = np.mean(gmm_labels == ds['Cluster'])


# In[27]:


#print
print("GMM Accuracy: %0.2f " % (gmm_accuracy))
print("GMM Execution Time : %0.4f"%(gmm_time))


# In[29]:


# Measure silhouette score
silhouette = silhouette_score(X, labels)
import psutil
print('Silhouette score:', round(silhouette,4))


# In[30]:


memory_usage = psutil.Process().memory_info().rss / 1024 ** 2  # in MB
print('Memory usage: The memory usage of the algorithm is approximately', memory_usage, 'MB.')


# In[31]:


#


# In[32]:


# Training Spectral Model


# In[33]:


spectral = SpectralClustering(n_clusters=3, assign_labels='discretize', random_state=42)
clusters = spectral.fit_predict(X_norm)


# In[34]:


ds['Cluster'] = clusters


# In[35]:


sns.scatterplot(x='HSLC', y='SSLC', hue='Cluster', size='Cgpa', data=ds)


# In[36]:


ds['Cluster'] = clusters
high_performing_students = ds[ds['Cluster'] == 2]
medium_performing_students = ds[ds['Cluster'] == 1]
low_performing_students = ds[ds['Cluster'] == 0]


# In[37]:


# Into excel sheet
writer = pd.ExcelWriter('Segmented_Model_SC_100_ds.xlsx')
high_performing_students.to_excel(writer, sheet_name='High Performing Students')
medium_performing_students.to_excel(writer, sheet_name='Medium Performing Students')
low_performing_students.to_excel(writer, sheet_name='Low Performing Students')
writer.save()


# In[38]:


# Calculating metrics for Comparision - SC


# In[39]:


#Execution Time
cluster = SpectralClustering(n_clusters=3, eigen_solver='arpack', affinity="nearest_neighbors")
start_time = time.time()
y_pred = cluster.fit_predict(X)
end_time = time.time()


# In[40]:


# Measure execution time
execution_time = end_time - start_time
print('Execution time:', round(execution_time,3), 'seconds')


# In[41]:


# Map cluster labels to performance levels
performance_levels = ['low', 'medium', 'high']
level_map = {0: 'low', 1: 'medium', 2: 'high'}
y_pred_performance = [level_map[i] for i in y_pred]


# In[43]:


# Evaluate accuracy
noise = np.random.normal(0, 0.1, size=X.shape)
X_noisy = X + noise
y_pred_noisy = cluster.fit_predict(X_noisy)
y_pred_performance_noisy = [level_map[i] for i in y_pred_noisy]
accuracy = accuracy_score(y_pred_performance, y_pred_performance_noisy)
print('Accuracy :', accuracy)


# In[45]:


# Evaluate clustering metrics
silhouette = silhouette_score(X, y_pred)
print('Silhouette score:', round(silhouette,4))


# In[46]:


# Memory Usage
memory_usage = psutil.Process().memory_info().rss / 1024 ** 2  # in MB
print('Memory usage: The memory usage of the algorithm is approximately', memory_usage, 'MB.')


# In[ ]:




