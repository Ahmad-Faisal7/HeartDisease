import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


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


elif section == "Model":
    st.header("Machine Learning Model: Logistic Regression")
    
    # Introduction to the Model Section
    st.write("""
    In this section, you can input your data to predict the likelihood of heart disease using a trained **Logistic Regression** model.
    Please fill in the following fields with your personal medical data, and the model will predict whether you are likely to have heart disease or not.
    """)

    # Input Fields for User Data
    with st.expander("Enter Your Data for Prediction", expanded=True):
        st.write("Please enter the required information below:")

        one_hot_columns = {'sex': ['Male', 'Female'], 'chest_pain_type': ['Typical angina', 'Atypical angina', 'Non-anginal pain', 'Asymptomatic'], 'fasting_blood_sugar': ['Lower than 120 mg/ml', 'Greater than 120 mg/ml'], 'rest_ecg': ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'],'exercise_induced_angina': ['Yes', 'No'],'slope': ['Up sloping', 'Down sloping', 'Flat'], 'vessels_colored_by_flourosopy': ['Zero', 'One', 'Two', 'Three', 'Four'], 'thalassemia': ['Reversable Defect', 'Fixed Defect', 'Normal', 'No']}

        # Collecting user inputs
        age = st.number_input("Age", min_value=1, max_value=120, step=1)
        sex = st.selectbox("Sex", options=["Male", "Female"])
        chest_pain_type = st.selectbox("Chest Pain Type", options=["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
        resting_blood_pressure = st.number_input("Resting Blood Pressure (mm Hg)", min_value=1, max_value=300, step=1)
        cholestoral = st.number_input("Cholesterol (mg/dl)", min_value=1, max_value=600, step=1)
        fasting_blood_sugar = st.selectbox("Fasting Blood Sugar", options=["Lower than 120 mg/ml", "Greater than 120 mg/ml"])
        rest_ecg = st.selectbox("Resting Electrocardiographic Results", options=["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
        max_heart_rate = st.number_input("Max Heart Rate Achieved", min_value=1, max_value=250, step=1)
        exercise_induced_angina = st.selectbox("Exercise Induced Angina", options=["Yes", "No"])
        oldpeak = st.number_input("Oldpeak (Depression Induced by Exercise Relative to Rest)", min_value=0.0, max_value=6.0, step=0.1)
        slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=["Up sloping", "Down sloping", "Flat"])
        vessels_colored_by_flourosopy = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=["Zero", "One", "Two", "Three", "Four"])
        thalassemia = st.selectbox("Thalassemia", options=["Reversable Defect", "Fixed Defect", "Normal", "No"])

        # Prepare the input data for prediction
        input_data = {
            'age': age,
            'sex': 1 if sex == "Male" else 0,  # Encoding sex
            'chest_pain_type': chest_pain_type,
            'resting_blood_pressure': resting_blood_pressure,
            'cholestoral': cholestoral,
            'fasting_blood_sugar': 1 if fasting_blood_sugar == "Greater than 120 mg/ml" else 0,  # Encoding fasting blood sugar
            'rest_ecg': rest_ecg,
            'max_heart_rate': max_heart_rate,
            'exercise_induced_angina': 1 if exercise_induced_angina == "Yes" else 0,  # Encoding exercise induced angina
            'oldpeak': oldpeak,
            'slope': slope,
            'vessels_colored_by_flourosopy': vessels_colored_by_flourosopy,
            'thalassemia': thalassemia
        }

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])
        # Apply one-hot encoding to the categorical variables
        for column, categories in one_hot_columns.items():
            # Apply one-hot encoding for each categorical column
            dummies = pd.get_dummies(input_df[column], prefix=column)
            input_df = pd.concat([input_df, dummies], axis=1)
            input_df.drop(column, axis=1, inplace=True)


        # Predict using the model
        if st.button("Predict Heart Disease"):
            prediction = model.predict(input_df)
            st.write(f"Prediction: {'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'}")
            st.write("Here’s a summary of your data:")
            st.write(input_df)

    # Display Model Accuracy and Other Metrics
    st.write("""
    Below, you'll find the performance of the trained model, including accuracy and evaluation metrics.
    """)
    with st.expander("Model Accuracy", expanded=True):
        accuracy = 0.81  # Precomputed accuracy
        st.write(f"The accuracy of the Logistic Regression model on the test set is **{accuracy:.2f}**.")
    
    with st.expander("Classification Report", expanded=False):
        st.write("""
        The classification report provides precision, recall, and F1-score metrics for both classes:
        - **Class 0 (No Disease)**: Patients without heart disease.
        - **Class 1 (Disease)**: Patients with heart disease.
        """)
        classification_report_text = """
              precision    recall  f1-score   support

           0       0.86      0.75      0.80       102
           1       0.78      0.88      0.83       103

    accuracy                           0.81       205
   macro avg       0.82      0.81      0.81       205
weighted avg       0.82      0.81      0.81       205
        """
        st.text(classification_report_text)
    
    # 3. Confusion Matrix
    with st.expander("Confusion Matrix", expanded=False):
        st.write("""
        The confusion matrix provides a breakdown of predictions:
        - **True Positives (TP)**: Correctly predicted as having heart disease.
        - **True Negatives (TN)**: Correctly predicted as not having heart disease.
        - **False Positives (FP)**: Incorrectly predicted as having heart disease.
        - **False Negatives (FN)**: Incorrectly predicted as not having heart disease.
        """)
        
        # Confusion Matrix Visualization
        conf_matrix = [[77, 25], [12, 91]]  # Example values from your results
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)



