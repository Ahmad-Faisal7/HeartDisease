import streamlit as st
import pandas as pd
import pickle

# Streamlit Application Title
st.title("🫀🩺 Heart Disease Application")

# Load the dataset
@st.cache  # Cache the dataset to improve performance
def load_data():
    return pd.read_csv("EDA and ML\HeartDisease.csv")

data = load_data()

# Load the trained model
@st.cache(allow_output_mutation=True)  # Cache the model to prevent reloading
def load_model():
    with open("EDA and ML\heart_disease_model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

# Introduction Section
with st.expander("Introduction", expanded=True):
    st.subheader("About the Project")
    st.write("""
    This project analyzes the **Heart Disease UCI Dataset** and builds a machine learning model 
    to predict the likelihood of heart disease based on patient data. The dataset includes features like:
    - Age, Gender, and Cholesterol Levels
    - Chest Pain Type and Resting Blood Pressure
    - Maximum Heart Rate Achieved and more.

    The app includes:
    1. **EDA Section:** Visualizations and insights to understand the dataset.
    2. **Model Section:** Predictions and evaluation of the trained machine learning model.
    3. **Conclusion:** Key takeaways from the analysis and results.

    Navigate through the app to explore the data, visualize patterns, and see predictions in action!
    """)
    st.info("💡 Tip: Use the sidebar to navigate through the app's sections.")

    # Display Dataset Preview
    if st.checkbox("Show Dataset Preview"):
        st.subheader("Dataset Preview")
        st.write(data.head())

    # Display Model Details
    if st.checkbox("Show Model Details"):
        st.subheader("Model Details")
        st.write("The model used in this app is a **Logistic Regression** classifier.")
        st.write("It was trained on the preprocessed data to achieve an accuracy of 81%.")

# Sidebar Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Introduction", "EDA", "Model", "Conclusion"])
