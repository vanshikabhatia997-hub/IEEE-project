import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="Iris Classifier", page_icon="🌸")


@st.cache_resource
def train_models():
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42).fit(X_train, y_train),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train),
    }
    return iris, X, models


iris, X, models = train_models()

st.title("🌸 Iris Flower Classifier")
st.write("Enter the four measurements (in cm) and the model predicts the species.")

# Sidebar: model choice
model_name = st.sidebar.selectbox("Choose a model", list(models.keys()))

# Sliders built from the real data ranges
inputs = {}
for col in X.columns:
    inputs[col] = st.slider(
        col.title(),
        min_value=float(X[col].min()),
        max_value=float(X[col].max()),
        value=float(X[col].mean()),
        step=0.1,
    )

if st.button("Predict species"):
    sample = pd.DataFrame([inputs])
    model = models[model_name]
    pred = model.predict(sample)[0]
    probs = model.predict_proba(sample)[0]

    st.success(f"Predicted species: **{iris.target_names[pred]}**")
    st.subheader("Confidence by species")
    st.bar_chart(pd.Series(probs, index=iris.target_names))