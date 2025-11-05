import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

#Loading all the pickle files and models

with open ("Label_Encoder_Gender.pkl","rb") as file:
    Label_Encoder_Gender = pickle.load(file)

with open ("OHE_Geography.pkl","rb") as file:
    OHE_Geography = pickle.load(file)

with open ("Standard_Scaler","rb") as file:
    Standard_Scaler = pickle.load(file)

model = load_model("ANN_model.h5")

#Streamlit App
st.title ("Churn Prediction App")

#User Input
geography = st.selectbox("Select the Geography", OHE_Geography.categories_[0])
gender = st.selectbox("Select the gender",Label_Encoder_Gender.classes_)
age = st.slider("Select the Age",18,100,25)
tenure = st.slider("Select the Tenure",0,20,3)
balance = st.number_input("Enter the Balance")
num_of_products = st.slider("Select the Number of Products",1,4,1)
credit_card = st.selectbox("Yes/No - Has Credit Card",[1,0])
active_member = st.selectbox("Yes/No - Is Active Member",[1,0])
estimated_salary = st.number_input("Enter the Estimated Salary")
credit_score = st.number_input("Enter the Credit Score")

#Input DataFrame
input_data = pd.DataFrame(data={'CreditScore':credit_score, 'Gender':Label_Encoder_Gender.transform([gender]), 'Age':age, 'Tenure':tenure, 
 'Balance':balance,'NumOfProducts':num_of_products, 'HasCrCard':credit_card, 'IsActiveMember':active_member, 'EstimatedSalary':estimated_salary})

#One Hot Encoding Geography
geography_array = OHE_Geography.transform([[geography]]).toarray()
df_geo = pd.DataFrame(geography_array,columns=OHE_Geography.get_feature_names_out())

#Final Input DataFrame
final_input_data =  pd.concat([input_data.reset_index(drop=True),df_geo],axis = 1)

#Scaling the data

final_input_data_scaled = Standard_Scaler.transform(final_input_data)

#Prediction

prediction = model.predict(final_input_data_scaled)
predict_proba = prediction[0][0]

if st.button("Predict"):
    if predict_proba > 0.5:
        st.error(f"The Customer will Churn with a probability of {predict_proba}")
    else:

        st.success(f"The Customer will not Churn with a probability of {1 - predict_proba}")
