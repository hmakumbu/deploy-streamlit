import streamlit as st
#import yfinance as yf
import pandas as pd
import numpy as np
#from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pickle
import time

# display
st.write(""" 
# Penguins Prediction
Data Obtained from the [palmerpenguins](https://github.com/dataprofessor/data/blob/master/penguins_cleaned.csv) 
""")

st.markdown('---')

# Display title in sidebar
st.sidebar.header('User Input Features')

st.sidebar.markdown("""[Example CSV input file](https://github.com/dataprofessor/data/blob/master/penguins_cleaned.csv)""")

with st.sidebar.expander('About App') :
    st.write('Here is the details of this app prediction, so this app can help you predict, the ............. blabalabakaa')

# Collect user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input csv", type=['csv'])

if uploaded_file is not None:
    progress_bar = st.progress(0)
    for per_completed in range(100):
        time.sleep(0.05)
        progress_bar.progress(per_completed + 1)
    
    try:
        input_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")
        st.stop()

    st.success('Your CSV file is well uploaded')
else:
    def user_input_features():
        island = st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
        sex = st.sidebar.selectbox('sex', ('male', 'female'))
        bill_length_mm = st.sidebar.slider('bill_length (mm)', 32.1, 59.6, 43.9)
        bill_depth_mm = st.sidebar.slider('bill_depth (mm)', 13.1, 21.5, 17.2)
        flipper_length_mm = st.sidebar.slider('flipper_length (mm)', 172.0, 231.0, 201.0)
        body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0, 6300.0, 4207.0) 
        data = {
            'island': island,
            'bill_length_mm': bill_length_mm,
            'bill_depth_mm': bill_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g,
            'sex': sex
        }
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

# Combine user features with entire penguins dataset
# This will be useful for the encoding phase
try:
    penguins_raw = pd.read_csv('penguins_cleaned.csv')
    penguins = penguins_raw.drop(columns=['species'])
    df = pd.concat([input_df, penguins], axis=0)
except FileNotFoundError:
    st.error("The penguins dataset file was not found. Please make sure the file 'penguins_cleaned.csv' is in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"Error processing the penguins dataset: {e}")
    st.stop()

# Encoding of ordinal features
encode = ['sex', 'island']
for col in encode:
    try:
        dummy = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummy], axis=1)
        df.drop(columns=[col], inplace=True)
    except Exception as e:
        st.error(f"Error encoding feature '{col}': {e}")
        st.stop()

df = df[:1]  # Select only the first row (the user input data)

st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be Uploaded. Currently using example input parameters')
    st.write(df)

# Load model
try:
    with open('penguins_clf.pkl', 'rb') as file:
        load_clf = pickle.load(file)
except FileNotFoundError:
    st.error("The model file 'penguins_clf.pkl' was not found. Please make sure the file is in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Apply model
try:
    prediction = load_clf.predict(df)
    prediction_proba = load_clf.predict_proba(df)
except Exception as e:
    st.error(f"Error making predictions: {e}")
    st.stop()

st.subheader('Prediction')
penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])

st.write(penguins_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
