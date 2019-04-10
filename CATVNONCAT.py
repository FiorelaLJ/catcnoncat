#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import h5py
 
    # Cargando los datos
train_dataset = h5py.File('E:/datasets/train_catvnoncat.h5', "r")
train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # train set features
train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # train set labels
 
test_dataset = h5py.File('E:/datasets/test_catvnoncat.h5', "r")
test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # test set features
test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # test set labels
 
classes = np.array(test_dataset["list_classes"][:]) # the list of classes
 
train_set_y = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
test_set_y = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))


# In[35]:


#mostrar uno de los datos
index = 14
plt.imshow(train_set_x_orig[index])
print("y = "+ str(train_set_y[:, index])+", it's a '"+classes[np.squeeze(train_set_y[:, index])].decode("utf-8")+"' picture.")


# In[36]:


#preprocesamiento
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
 
print ("Dataset dimensions:")
print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))


# In[37]:


#redimensionando los datos
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T 
 
print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))


# In[38]:


#normalizacion de datos
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.


# In[39]:


#inicializando variables
def initialize_with_zeros(dim):
    assert(w.shape==(dim,1))
    assert(isinstance(b,float) or isinstance(b, int))
    return w,b


# In[40]:


#funcion sigmoide
def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s


# In[ ]:




