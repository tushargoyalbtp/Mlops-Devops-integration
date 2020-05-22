#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('wines.csv')


# In[3]:


df.info()


# In[4]:


y = df['Class']


# In[5]:


y.value_counts()


# In[6]:


y_cat = pd.get_dummies(y)


# In[7]:


y


# In[ ]:





# In[8]:


df.columns


# In[9]:


X = df.drop('Class' , axis=1)


# In[10]:


X.info()


# In[11]:


import seaborn as sns


# In[12]:


sns.scatterplot(x='Alcohol' , y=y , data=df)


# In[13]:


from keras.models import Sequential


# In[14]:


model  =  Sequential()


# In[18]:


X.info()


# In[39]:


X.shape


# In[40]:


y_cat.shape


# In[16]:


from keras.layers import Dense


# In[20]:


model.add(Dense(units=5 , input_shape=(13,), 
                activation='relu', 
                kernel_initializer='he_normal' ))


# In[21]:


model.summary()


# In[22]:


model.add(Dense(units=8 , 
                activation='relu', 
                kernel_initializer='he_normal' ))


# In[23]:


model.summary()


# In[24]:


model.add(Dense(units=2, 
                activation='relu', 
                kernel_initializer='he_normal' ))


# In[25]:


model.summary()


# In[26]:


model.add(Dense(units=3, activation='softmax'))


# In[27]:


model.summary()


# In[28]:


from keras.optimizers import RMSprop


# In[30]:


model.compile(optimizer=RMSprop(learning_rate=0.01),  
              loss='categorical_crossentropy',
             metrics=['accuracy']
             )


# In[34]:


model.layers[0].input


# In[37]:


model.layers[3].output


# In[38]:


model.layers[2].output


# In[45]:


model.fit(X,y_cat, epochs=100)


# In[43]:


# import keras.backend as K


# In[44]:


# K.clear_session()


# In[46]:


model.get_weights()


# In[47]:


model.save('modelsave.h5')


# In[ ]:




