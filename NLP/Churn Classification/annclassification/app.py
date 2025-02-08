import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder
import pickle 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import datetime
from tensorflow.keras.models import load_model

model = load_model('model.h5')

with open('one_hot_encoder_geo.pkl','rb') as file : 
    geo_encoder = pickle.load(file)
with open('label_encoder_gender.pkl','rb') as file : 
    gender_encoder = pickle.load(file)
with open('scalar.pkl','rb') as file : 
    scalar = pickle.load(file)


st.title('Employee Churn Predictor')

credit = st.number_input('Enter the Credit Score')
geography = st.selectbox('Country: ', geo_encoder.categories_[0])
gender = st.selectbox('Select your Gender',gender_encoder.classes_)
age = st.slider('Select your Age',18,100)
st.text('Age: {}'.format(age))
estimatedSalary = st.number_input('Enter the estimated Salary')
tenure = st.slider('Select Tenure at Company',0,10)
st.text('Tenure: {}'.format(tenure))
balance = st.number_input('Enter the Balance amount')
numberOfProducts = st.slider('Select the number of products',1,10)
st.text('Products: {}'.format(numberOfProducts))
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Are you an active member? ',[0,1])


created_dict = {
    'CreditScore': [credit],
    'Geography': [geography],
    'Gender' : [gender_encoder.transform([gender])[0]],
    'Age' : [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts' :[numberOfProducts],
    'HasCrCard':[has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary':[estimatedSalary]   
}

input_data = pd.DataFrame(created_dict)

encoded_geography = geo_encoder.transform([[geography]]).toarray()

geo_encoded_df = pd.DataFrame(encoded_geography, columns=geo_encoder.get_feature_names_out())

input_data_df = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)
input_data_df = input_data_df.drop('Geography', axis=1)


scaled_data = scalar.transform(input_data_df)

prediction = model.predict(scaled_data)

if prediction < 0.5:
    st.success("The employee is most likely to not churn")
else:
    st.error("The employee is likely to churn")