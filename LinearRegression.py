#!/usr/bin/env python
# coding: utf-8

# ### Import all libraries

# In[1]:


from pyspark import SparkContext
from pyspark.sql import *
from pyspark.sql.functions import col
import numpy as np
import sys
from pyspark.sql import functions as F


# ### Create Spark Context

# In[2]:


sc = SparkContext()
spark = SparkSession.builder.master("local").appName("LinearRegression").config("spark.some.config.option", "some-value").getOrCreate()


# ### Data Processing
# - An index is introduced to the dataframe so that the input and output can be paired correctly
# - The input is taken in a RDD X and it has been added with 1 to take into consideration the bias
# - The output is taken in a RDD Y 

# In[4]:


# df = spark.read.csv("yxlin.csv",inferSchema =True,header=False)
df = spark.read.csv(sys.argv[1],inferSchema =True,header=False)


# In[5]:


df=df.withColumn("idx",F.monotonically_increasing_id())


# In[6]:


data = df.select(col("_c0").alias("Y"), col("_c1").alias("X"), col("idx").alias("Index"))
X = data.select(["Index","X"]).rdd.map(lambda x : (x[0], np.vstack((1,np.array((x[1]))))))
Y = data.select(["Index","Y"]).rdd.map(lambda x : (x[0],np.array(([x[1]]))))


# - Persist the RDDs to make them reusable for future use

# In[7]:


X.persist()
Y.persist()


# - Calculate the number of features and the number of samples in the dataset

# In[8]:


features=X.collect()[0][1].shape[0]
samples= X.count()


# - Join the X and Y rdds which gives a tuple of index and input output pair

# In[9]:


x_exp_y = X.join(Y)


# In[11]:


x_exp_y.persist()


# ### Linear Regression
# - emit_A(): Takes features and calculates the product with its transpose
# - emit_B(): Takes a pair of features and output labels and returns the product of their pair
# - weights(): Calculates the weights for this linear regression model. A is calculated by reducing the keys from emit_A. Its inverse is then calculated. B is calculated by reducing the keys from emit_B. A_inv and B are then joined and multiplied to calculate the final weights.

# In[12]:


def emit_A(x):
    A = np.dot(x,x.T)
    key = "key"
    return key, A


# In[13]:


def emit_B(x):
    B = np.dot(x[0],x[1])
    key = "key"
    return key, B


# In[14]:


def weights():
    A = X.map(lambda x: emit_A(x[1]))
    A = A.reduceByKey(lambda x,y: np.add(x, y))
    A_inv = A.map(lambda x: (x[0],np.linalg.inv(x[1])))
    B = x_exp_y.map(lambda x: emit_B(x[1]))
    B = B.reduceByKey(lambda x,y: np.add(x, y))
    beta = A_inv.join(B)
    final_weights = beta.map(lambda x: np.dot(x[1][0],x[1][1]))
    return final_weights


# In[15]:


weightsrdd=weights()
weights = weightsrdd.collect()


# ### Linear Regression with Gradient Descent
# - fun_dw(): Takes input as a pair of features and output labels and calculates the predicted label. The derivative of the cost function MSE is calculated by ignoring the scaling factor.
# 
# \begin{split}f'(w) =
#    \begin{bmatrix}
#      \frac{df}{dw}\\
#     \end{bmatrix}
# =
#    \begin{bmatrix}
#      \frac{1}{N} \sum 2x_i((wx_i)-y_i) \\
#     \end{bmatrix}\end{split}

# In[21]:


def fun_dw(xy):
    y_pred = np.dot(weights_init,xy[0])
    dw=np.dot(xy[0],(y_pred-xy[1]))
    return dw


# - The below algorithm is run for 50 iterations and a learning rate of 0.1. The weights are initialzied to 0. The summation of all dw components are calculated according to the above formula and the weights are updated.

# In[42]:

weights_init = np.zeros(features)
for i in range(50):
  dw = x_exp_y.map(lambda xy: ("dw", fun_dw(xy[1]))).reduceByKey(lambda x,y: x+y).map(lambda x:x[1]*1/samples)
  weights_init -= 0.01*dw.collect()[0]

weights_gd = weights_init
# In[16]:


print("The linear coefficients of the linear regression model are:")
print("-"*50)
print(weights)


# In[51]:


print("The linear coefficients using GRADIENT DESCENT(learning rate = 0.01, iterations = 50) are:")
print("-"*50)
print(weights_gd)


# In[ ]:


weightsrdd.saveAsTextFile(sys.argv[2])


# In[55]:




# In[ ]:


X.unpersist()
Y.unpersist()
x_exp_y.unpersist()


# In[ ]:


# regression_line = [(weights[0][1][1]*x)+weights[0][1][0] for x in x_plot]


# In[ ]:


# import matplotlib.pyplot as plt
# from matplotlib import style
# style.use('ggplot')


# In[ ]:


# plt.scatter(x_plot,y_plot,color='black')
# plt.plot(x_plot, regression_line, color='red')
# plt.show()

