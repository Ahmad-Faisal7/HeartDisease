# 🫀🩺 Heart Disease Prediction App
```

```



[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://HeartDisease.streamlit.app/)

## GitHub Codespaces

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/app-starter-kit?quickstart=1)

This repository contains a machine learning project that predicts the likelihood of heart disease based on various health indicators. The project uses exploratory data analysis (EDA), machine learning models, and a Streamlit app for user interaction.

## Project Structure

- **eda_and_ml**: Contains Jupyter Notebooks for exploratory data analysis (EDA) and model training. It includes code to preprocess the dataset, visualize insights, and train a machine learning model.
- **trained_model.pkl**: A serialized (pickled) file containing the trained machine learning model that predicts heart disease likelihood.
- **dataset**: The dataset used for training the model, which contains various health-related features of patients.
- **streamlit_app**: A Streamlit web application for interacting with the trained model. It allows users to input health data and predict the likelihood of heart disease.

## Project Overview

The goal of this project is to create a machine learning model capable of predicting whether a person is likely to have heart disease based on various health indicators such as age, cholesterol levels, blood pressure, and more.

The project follows these main steps:
1. **Exploratory Data Analysis (EDA)**: Understanding the dataset, cleaning the data, and visualizing trends.
2. **Model Training**: Training a machine learning model using algorithms like logistic regression or decision trees.
3. **Model Evaluation**: Evaluating the model's performance using accuracy, precision, recall, and F1-score metrics.
4. **Streamlit App**: A web interface that uses the trained model to predict heart disease likelihood based on user inputs.

## How to Run the Streamlit App

To run the Streamlit app and make predictions, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Ahmad-Faisal7/HeartDisease.git
    cd HeartDisease
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run streamlit_app/app.py
    ```

4. The app will open in your browser, where you can input health data and see predictions for heart disease.

## Model Performance

- **Accuracy**: The model achieves an accuracy of 81%, correctly predicting the presence or absence of heart disease 81% of the time.
- **Precision & Recall**: The model performs well in identifying both patients with and without heart disease.
    - For patients **without heart disease (class 0)**: 75% correctly predicted.
    - For patients **with heart disease (class 1)**: 88% correctly predicted.

## Future Improvements

- The model can be further improved by incorporating more data and experimenting with advanced machine learning techniques.
- A more user-friendly web interface and additional features could be added to the Streamlit app.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
