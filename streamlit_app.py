import streamlit as st

# Streamlit Application Title
st.title("🫀🩺 Heart Disease Prediction Application")

# Introduction Section
with st.expander("Introduction", expanded=True):
    st.subheader("About the Project")
    st.write("""
    This project aims to analyze the **Heart Disease UCI Dataset** and build a machine learning model 
    to predict the likelihood of heart disease based on patient data. The dataset contains features such as:
    - Age, Gender, and Cholesterol Levels
    - Chest Pain Type and Resting Blood Pressure
    - Maximum Heart Rate Achieved and more.

    The project is divided into the following sections:
    1. **Exploratory Data Analysis (EDA):** Insights and visualizations to understand the dataset.
    2. **Machine Learning Model:** Training and evaluation of a classification model.
    3. **Conclusion:** Highlights of the project and its findings.

    Navigate through the app to explore the data, visualize patterns, and see predictions in action!
    """)
    st.info("💡 Tip: Use the sidebar to navigate through the app's sections.")

# Navigation Section
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Introduction", "EDA", "Model", "Conclusion"])
