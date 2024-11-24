
import streamlit as st
import numpy as np
import pandas as pd
import pickle 

#Load the instamces that were created
with open('final_model.pkl','rb') as file:
    model=pickle.load(file)

with open('pca.pkl','rb') as file:
    pca=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

def prediction (input_data):
    scaled_data=scaler.transform(input_data)
    pca_data= pca.transform(scaled_data)
    pred=model.predict(pca_data)[0]

    if pred==0:
        return 'Developing'
    elif pred==1:
        return 'Underdeveloped'
    else:
        return 'Developed'

def main():
    st.title('HELP International Foundation')
    st.subheader('This application will give the status of the country based on socio-economic and health factors')
    ch_mort=st.text_input('Enter Child Mortality Rate:')
    exp=st.text_input('Enter Exports (% GDP):')
    hel=st.text_input('Enter Expenditure on Health (% GDP):')
    imp=st.text_input('Enter Imports (% GDP):')
    income=st.text_input('Enter Average Net Income:')
    inf=st.text_input('Enter Inflation Rate:')
    life_exp=st.text_input('Enter Average Life Expectancy:')
    fer=st.text_input('Enter Fertility Rate:')
    gdp=st.text_input('Enter GDP per population:')

    input_list=[[ch_mort,exp,hel,imp,income,inf,life_exp,fer,gdp]]

    if st.button('predict'):
        response=prediction(input_list)
        st.success(response)

if __name__=='__main__':
    main()
