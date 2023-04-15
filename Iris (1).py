#!/usr/bin/env python
# coding: utf-8

# # Data Science & Business Analytics internship at TSF Group

# # The Sparks Foundation

# Task 2- Predict the optimum number of clusters and represent it visually.
# 
# Prediction using Unsupervised ML 
# 
# By - Vishal Kobarne Data Science & Business Analytics intern at The Sparks Foundation (TSF)

# # Importing the required Libraries
# 

# In[52]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')
print('Libraries are imported Successfully..')


# #  Importing the dataset

# In[34]:


df = pd.read_csv(r"C:\Users\Kiran\Downloads\Iris (1).csv")
print("Iris data set read successfully...")


# In[35]:


df.head()


# In[36]:


df.tail()


# # Checking the shape of data
# 

# In[37]:


df.shape


# # Full Data Summary
# 

# In[38]:


df.info()


# # Statistical Summary of Data

# In[39]:


df.describe()


# # Columns in the data

# In[40]:


df.columns


# #  Finding out Null values in Each Columns

# In[41]:


df.isnull().sum()


# In[42]:


df.isna().sum()


# -In this dataset  we can see there is no null or missing value contain 

# #  Plotting the results into a line graph, 

# In[43]:


x = df.iloc[:, [0, 1, 2, 3]].values


# In[44]:


from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # Within cluster sum of squares
plt.show()


# As we can see, the elbow is located at around 3 clusters. Therefore, we can conclude that the optimum number of clusters for this dataset is 3.

# #  Applying kmeans to the dataset / Creating the kmeans classifier
# 

# In[45]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# # Visualising the clusters and  plotting the centroids of the clusters
# 

# In[51]:


plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 100, c = 'Yellow', label = 'Iris-virginica')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'Black', label = 'Centroids')
plt.xlabel('Sepal length',fontsize = 15)
plt.ylabel('Sepal width',fontsize = 15)
plt.title('Cluster of Iris data')

plt.legend()


# We can see that the three clusters are fairly distinct and well-separated.
