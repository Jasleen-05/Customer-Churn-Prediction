import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import warnings
warnings.filterwarnings("ignore")

# App Title
st.title("📉 Customer Churn Prediction App")
st.markdown("Predict whether a customer will churn based on their demographic and service usage information.")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Raw Dataset")
    st.dataframe(df.head())

    # Drop columns
    if "TotalCharges" in df.columns:
        df.drop("TotalCharges", axis=1, inplace=True)
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].apply(lambda x: 'No' if x == 0 else 'Yes')

    # Show info
    st.subheader("🧾 Data Info")
    st.write(df.info())

    # Visualizations
    st.subheader("📈 Exploratory Data Analysis")
    num_vars = [col for col in df.columns if df[col].dtype != 'object']
    fig, ax = plt.subplots(1, len(num_vars), figsize=(15, 5))
    for i, col in enumerate(num_vars):
        ax[i].hist(df[col][df.Churn == 'No'], label='No', bins=30, alpha=0.7)
        ax[i].hist(df[col][df.Churn == 'Yes'], label='Yes', bins=30, alpha=0.7)
        ax[i].set_title(col)
        ax[i].legend()
    st.pyplot(fig)

    # Categorical Plots
    st.subheader("📊 Categorical Variables vs Churn")
    cat_vars = [col for col in df.columns if df[col].dtype == 'object' and col != 'Churn']
    fig, axs = plt.subplots(4, 4, figsize=(20, 20))
    for ax, col in zip(axs.flat, cat_vars):
        sns.countplot(x='Churn', hue=col, data=df, ax=ax)
    st.pyplot(fig)

    # Preprocessing
    le = LabelEncoder()
    for col in df.columns:
        if len(df[col].unique()) == 2:
            df[col] = le.fit_transform(df[col])

    df = pd.get_dummies(df, columns=[col for col in df.columns if df[col].dtype == 'object'], drop_first=True)

    # Train/Test Split
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # Models
    st.subheader("🧠 Model Results")

    def evaluate_model(name, model, X_train, y_train, X_test, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.markdown(f"**{name} Accuracy:** {acc * 100:.2f}%")
        st.text(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        st.pyplot(fig)

    # Logistic Regression
    st.markdown("### 📌 Logistic Regression")
    evaluate_model("Logistic Regression", LogisticRegression(), X_train, y_train, X_test, y_test)

    # Decision Tree
    st.markdown("### 🌳 Decision Tree")
    evaluate_model("Decision Tree", DecisionTreeClassifier(), X_train, y_train, X_test, y_test)

    # KNN
    st.markdown("### 🤝 K-Nearest Neighbors")
    evaluate_model("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=5), X_train, y_train, X_test, y_test)

    # SVM
    st.markdown("### 🧮 Support Vector Machine")
    evaluate_model("SVM", SVC(kernel='linear', random_state=42), X_train, y_train, X_test, y_test)

else:
    st.info("Please upload a dataset (`data2_file.csv`) to begin.")