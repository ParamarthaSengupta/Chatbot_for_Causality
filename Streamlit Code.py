import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from Hackathon_Utilities_v1 import *
from PIL import Image

import time
st.cache_data.clear()
    
    # Clears all cached resource functions
st.cache_resource.clear()
# Function to simulate a 3-minute countdown timer

st.set_page_config(page_title="Chatbot for Causality", layout="wide")

# Title of the app
st.title("Chatbot for Causality")

# Step 1: Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Step 2: Read the CSV file
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(data.head())

    # Step 3: Select the type of problem
    problem_type = st.selectbox(
        "Select the type of problem you want to solve",
        ["Classification","Regression"]
    )

    # Step 4: Input the question the user wants to ask
    user_question = st.text_input("What question would you like to ask about the data?")

    # Step 5: Process based on problem type
    if st.button("Run Analysis"):
        st.write(f"Selected problem: {problem_type}")
        st.write(f"Question: {user_question}")
        # countdown(180)
        # st.empty()
        with st.spinner("Running the analysis... Please wait."):
            

            result = main(
                data_source=data, 
                data_type='csv', 
                task=problem_type,
                ask=user_question
            )
            st.write(result)

        
