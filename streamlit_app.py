import streamlit as st
import pandas as pd
import pickle

# Streamlit Application Title
st.title("🫀🩺 Heart Disease Application")

# Load the dataset
@st.cache_data  # Cache the dataset to improve performance
def load_data():
    return pd.read_csv("EDA and ML/HeartDisease.csv")

data = load_data()

# Load the trained model
@st.cache_resource # Cache the model to prevent reloading
def load_model():
    with open("EDA and ML/heart_disease_model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

# Sidebar Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Introduction", "EDA", "Model", "Conclusion"])

# Introduction Section
if section == "Introduction":
    st.header("Introduction")
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

elif section == "EDA":
    st.header("Exploratory Data Analysis (EDA)")
    
    # 1. Histograms
    with st.expander("Visualizations: Histograms", expanded=False):
        st.write("Histograms of numerical features to understand their distributions.")
        fig, ax = plt.subplots(figsize=(12, 10))
        data.hist(ax=ax, bins=20)
        plt.suptitle('Histograms of Numerical Features', fontsize=16)
        st.pyplot(fig)
    
    # 2. Correlation Analysis
    with st.expander("Correlation Analysis", expanded=False):
        st.write("Correlation heatmap showing relationships between numerical features.")
        
        # Encode gender column (example for demonstration)
        data_encoded = data.copy()
        if 'Gender' in data_encoded.columns:
            data_encoded['Gender'] = data_encoded['Gender'].map({'Male': 0, 'Female': 1})
        
        numeric_data = data_encoded.select_dtypes(include=['number'])
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        plt.title('Correlation Heatmap', fontsize=14)
        st.pyplot(fig)
    
    # 3. Missing Value Analysis
    with st.expander("Missing Value Analysis", expanded=False):
        st.write("Analysis of missing values in the dataset.")
        missing_values = data.isnull().sum()
        missing_percentage = (missing_values / len(data)) * 100
        st.write("Missing Values and Percentages:")
        st.write(pd.DataFrame({
            'Missing Count': missing_values,
            'Percentage': missing_percentage
        }))
    
    # 4. Outlier Detection
    with st.expander("Outlier Detection", expanded=False):
        st.write("Boxplots for numerical features to detect potential outliers.")
        numerical_cols = ['age', 'cholestoral', 'resting_blood_pressure', 'Max_heart_rate']
        for col in numerical_cols:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(x=data[col], ax=ax)
            plt.title(f'Boxplot of {col}', fontsize=14)
            st.pyplot(fig)
    
    # 5. Feature Distribution Analysis
    with st.expander("Feature Distribution Analysis", expanded=False):
        st.write("Distribution of features by target variable.")
        
        # Age distribution
        st.write("Age Distribution by Heart Disease")
        fig, ax = plt.subplots()
        sns.kdeplot(data=data, x='age', hue='target', fill=True, ax=ax)
        plt.title('Age Distribution by Heart Disease', fontsize=14)
        st.pyplot(fig)
        
        # Cholesterol distribution
        st.write("Cholesterol Distribution by Heart Disease")
        fig, ax = plt.subplots()
        sns.kdeplot(data=data, x='cholestoral', hue='target', fill=True, ax=ax)
        plt.title('Cholesterol Distribution by Heart Disease', fontsize=14)
        st.pyplot(fig)
    
    # 6. Data Types and Unique Value Counts
    with st.expander("Data Types and Unique Value Counts", expanded=False):
        st.write("Overview of data types and unique values in the dataset.")
        data_types = data.dtypes
        unique_counts = data.nunique()
        st.write(pd.DataFrame({
            'Data Type': data_types,
            'Unique Values': unique_counts
        }))
    
    # 7. Trend Analysis
    with st.expander("Trend Analysis", expanded=False):
        st.write("Scatter plot showing the relationship between age and cholesterol levels.")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=data, x='age', y='cholestoral', hue='target', palette='coolwarm', ax=ax)
        plt.title('Age vs. Cholesterol by Heart Disease', fontsize=14)
        st.pyplot(fig)
    
    # 8. Grouped Aggregations
    with st.expander("Grouped Aggregations", expanded=False):
        st.write("Mean statistics of features grouped by the target variable.")
        numeric_cols = data.select_dtypes(include=['number']).columns
        grouped_stats = data[numeric_cols].groupby(data['target']).mean()
        st.write("Grouped Mean Statistics by Target:")
        st.write(grouped_stats)
        
        st.write("Visualization of grouped mean statistics:")
        fig, ax = plt.subplots(figsize=(10, 6))
        grouped_stats[['age', 'cholestoral', 'resting_blood_pressure']].plot(kind='bar', ax=ax)
        plt.title('Mean Values Grouped by Target', fontsize=14)
        plt.ylabel('Mean Value')
        plt.xticks(rotation=0)
        plt.grid(True)
        st.pyplot(fig)
    
    # 9. Pairwise Relationships
    with st.expander("Pairwise Relationships", expanded=False):
        st.write("Pairplot showing relationships between selected numerical features.")
        fig = sns.pairplot(data, vars=['age', 'cholestoral', 'resting_blood_pressure', 'Max_heart_rate'], hue='target', palette='coolwarm', diag_kind='kde')
        fig.fig.suptitle('Pairwise Relationships of Numerical Features', y=1.02, fontsize=16)
        st.pyplot(fig)




