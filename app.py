# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

@st.cache_data
def load_data():
    return pd.read_csv('data/Titanic-Dataset.csv')

@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

def show_data_overview(df):
    st.subheader("Dataset overview")
    st.write(f"Shape: {df.shape}")
    st.write(df.dtypes)
    if st.checkbox("Show raw data"):
        st.dataframe(df.sample(100))

def show_visualizations(df):
    st.subheader("Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Survival by Sex**")
        # Fix column name case sensitivity here, titanic dataset often uses 'sex' lowercase
        fig = px.histogram(df, x='Sex', color='Survived', barmode='group', labels={'Survived':'Survived (0=No, 1=Yes)'})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("**Fare distribution**")
        fig2 = px.histogram(df, x='Fare', nbins=50)
        st.plotly_chart(fig2, use_container_width=True)
    st.markdown("**Age vs Fare (colored by Survived)**")
    fig3 = px.scatter(df, x='Age', y='Fare', color='Survived', hover_data=['Pclass','Sex'])
    st.plotly_chart(fig3, use_container_width=True)

def model_predict_ui(model):
    st.subheader("Make a prediction")
    with st.form("predict_form"):
        pclass = st.selectbox("Pclass", options=[1,2,3], index=2)
        sex = st.selectbox("Sex", options=['male','female'])
        age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
        sibsp = st.number_input("SibSp", min_value=0, max_value=10, value=0)
        parch = st.number_input("Parch", min_value=0, max_value=10, value=0)
        fare = st.number_input("Fare", min_value=0.0, max_value=1000.0, value=10.0)
        embarked = st.selectbox("Embarked", options=['S', 'C', 'Q'])
        submitted = st.form_submit_button("Predict")
    if submitted:
        try:
            # Use lowercase feature names and encode categorical features
            input_df = pd.DataFrame([{
                'pclass': pclass,
                'sex': 0 if sex == 'male' else 1,
                'age': age,
                'sibsp': sibsp,
                'parch': parch,
                'fare': fare,
                'embarked': {'S': 0, 'C': 1, 'Q': 2}[embarked]
            }])
            pred = model.predict(input_df)[0]
            st.write("**Prediction:**", "Survived ✅" if pred==1 else "Did not survive ❌")
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_df)[0][1]
                st.write(f"**Survival probability:** {proba:.2f}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

def show_model_performance(model, df):
    st.subheader("Model performance (on held-out test set)")
    from sklearn.model_selection import train_test_split
    target = 'Survived'
    features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
    # Lowercase columns to match training
    df_lower = df.rename(columns={col: col.lower() for col in df.columns})
    X = df_lower[[f.lower() for f in features]].copy()
    y = df_lower[target.lower()]
    # Encode categorical features as during training
    X['sex'] = X['sex'].map({'male': 0, 'female': 1})
    X['embarked'] = X['embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    # Fill any missing values if needed
    X = X.fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    st.write(cm)

def main():
    st.title("Titanic Survival Prediction App")
    df = load_data()
    model = load_model()
    show_data_overview(df)
    show_visualizations(df)
    model_predict_ui(model)
    show_model_performance(model, df)

if __name__ == "__main__":
    main()
