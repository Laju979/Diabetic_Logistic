
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.header('Diabetes Prediction Created by Perfection')

datafile = pd.read_csv('Dataset of Diabetes .csv')
df = datafile[['Gender', 'AGE', 'Urea', 'Cr', 'HbA1c', 'Cholesterol', 'TG', 'HDL', 'LDL', 'VLD', 'BMI', 'CLASS']]

encoder_Gender = LabelEncoder()
df['Gender'] = encoder_Gender.fit_transform(df['Gender'])

f = df[['Gender', 'AGE', 'Urea', 'Cr', 'HbA1c', 'Cholesterol', 'TG', 'HDL', 'LDL', 'VLD', 'BMI']]
c = df[['CLASS']]

feature_train, feature_test, target_train, target_test = train_test_split(f, c, test_size=0.3)

model = LogisticRegression()
model.fit(feature_train, target_train)


st.sidebar.header('DIABETES CHECK')
NAME = st.sidebar.text_input('Name')
AGE = st.sidebar.slider('Age', 0, 100)
Gender = st.sidebar.selectbox('Gender', encoder_Gender.classes_)
Urea = st.sidebar.number_input('Urea', min_value=0)
Cr = st.sidebar.number_input('Cr', min_value=0)
HbA1c = st.sidebar.number_input('HbA1c', min_value=0)
Cholesterol = st.sidebar.number_input('Cholesterol', min_value=0)
TG = st.sidebar.number_input('TG', min_value=0)
HDL = st.sidebar.number_input('HDL', min_value=0)
LDL = st.sidebar.number_input('LDL', min_value=0)
VLD = st.sidebar.number_input('VLD', min_value=0)
BMI = st.sidebar.number_input('BMI', min_value=0)

ENC = encoder_Gender.transform([Gender])[0]

columns = {'Gender': [ENC],
           'AGE': [AGE],
           'Urea': [Urea],
           'Cr': [Cr],
           'HbA1c': [HbA1c],
           'Cholesterol': [Cholesterol],
           'TG': [TG],
           'HDL': [HDL],
           'LDL': [LDL],
           'VLD': [VLD],
           'BMI': [BMI]}


input_features = pd.DataFrame(columns)
st.write('Personal Details:', input_features)

st.write('The class category is based on Y, N, P.')
st.write(' Y means the person is Diabetic')
st.write('N means the person is non-diabetic')
st.write('while P means the person is predicted-diabetic')


if st.button('Diagnosis'):
    prediction = model.predict(input_features)
    #st.write('Our diagnosis based on what you have given us is', prediction)
    st.write(f'Our diagnosis based on what you have given us for {NAME} is: {prediction[0]}')

