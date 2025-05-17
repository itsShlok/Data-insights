#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install altair==4.2.2


# In[2]:


import pandas as pd
import numpy as np
import streamlit as st
import pickle
import joblib
import warnings
warnings.filterwarnings('ignore')


# In[3]:


#!pip install altair --upgrade


# In[4]:


#!pip install pandas scikit-learn numpy altair


# In[5]:


#!pip install streamlit


# In[32]:


# Load model and scaler
model = pickle.load(open('Dep.pkl', 'rb'))
scaler = joblib.load('scaler.pkl')


# In[34]:


# Streamlit UI
st.title('Model Deployment Using Logistic Regression')


# In[36]:


pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ['male', 'female'])
age = st.slider("Age", 0, 100, 25)
fare = st.slider("Fare", 0, 600, 50)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 8, 0)
parch = st.number_input("Number of Parents/Children Aboard", 0, 6, 0)
embarked = st.selectbox("Port of Embarkation", ['C', 'Q', 'S'])


# In[38]:


# Mapping sex to dummy variable
sex_male = 1 if sex == 'male' else 0


# In[40]:


# Mapping embarked to dummy variables
embarked_Q = 1 if embarked == 'Q' else 0
embarked_S = 1 if embarked == 'S' else 0


# In[42]:


# Create input data in the correct order
input_data = np.array([[pclass, age, sibsp, parch, fare, sex_male, embarked_Q, embarked_S]])


# In[12]:


input_data_scaled = scaler.transform(input_data)


# In[44]:


# Scale input
input_data_scaled = scaler.transform(input_data)


# In[46]:


# Prediction
prediction = model.predict(input_data_scaled)[0]
probability = model.predict_proba(input_data_scaled)[0][1]


# In[48]:


# Display output
st.subheader("Prediction")
st.write("🎯 Survived" if prediction == 1 else "💀 Did Not Survive")
st.write(f"Survival Probability: {probability:.2f}")


# In[ ]:





# In[ ]:




