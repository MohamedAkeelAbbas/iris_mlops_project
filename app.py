"""
Iris Classification - Streamlit App
Deploy ML model with interactive UI
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(page_title="Iris Classifier", page_icon="🌸", layout="wide")

# Title
st.title("🌸 Iris Flower Classification")
st.markdown("**MLOps Demo** - Predict Iris flower species using Machine Learning")

# Sidebar
st.sidebar.header("Model Configuration")


@st.cache_resource
def load_model():
    """Train a fresh model for reliable cloud deployment"""
    iris = load_iris()
    X_train, _, y_train, _ = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Store scaler with model for prediction
    model.scaler = scaler

    return model, "Model ready"


# Load model
model, model_status = load_model()
st.sidebar.success(model_status)

# Class names
class_names = ["Setosa", "Versicolor", "Virginica"]

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Input Features")

    # Input sliders
    sepal_length = st.slider(
        "Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.8, step=0.1
    )
    sepal_width = st.slider(
        "Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.0, step=0.1
    )
    petal_length = st.slider(
        "Petal Length (cm)", min_value=1.0, max_value=7.0, value=4.0, step=0.1
    )
    petal_width = st.slider(
        "Petal Width (cm)", min_value=0.1, max_value=2.5, value=1.2, step=0.1
    )

with col2:
    st.subheader("🎯 Prediction")

    # Prepare input
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Scale if scaler exists
    if hasattr(model, "scaler"):
        input_scaled = model.scaler.transform(input_data)
    else:
        input_scaled = input_data

    # Predict
    if st.button("🔮 Predict Species", type="primary"):
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]

        predicted_class = class_names[prediction]

        # Display result
        st.success(f"### Predicted: **{predicted_class}**")

        # Show probabilities
        st.markdown("**Confidence:**")
        for i, (name, prob) in enumerate(zip(class_names, probabilities)):
            st.progress(prob, text=f"{name}: {prob:.1%}")

        # Show flower emoji based on prediction
        emojis = ["🌺", "🌼", "🌷"]
        st.markdown(f"## {emojis[prediction]}")

# Dataset info
st.markdown("---")
st.subheader("📈 Dataset Overview")

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = [class_names[i] for i in iris.target]

col3, col4 = st.columns(2)

with col3:
    st.markdown("**Sample Data:**")
    st.dataframe(df.head(10), use_container_width=True)

with col4:
    st.markdown("**Statistics:**")
    st.dataframe(df.describe(), use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit |
        <a href='https://github.com/VenkateswarluPudur/mlops'>GitHub Repo</a></p>
    </div>
    """,
    unsafe_allow_html=True,
)
