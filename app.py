import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix
)
import matplotlib.pyplot as plt

# ---------------------------------
# PAGE CONFIG
# ---------------------------------

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

# ---------------------------------
# LOAD CSS
# ---------------------------------

with open("style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# ---------------------------------
# LOAD DATA
# ---------------------------------

df = pd.read_csv(
    "Titanic-Dataset.csv"
)

df["Age"] = df["Age"].fillna(
    df["Age"].mean()
)

features = [
    "Pclass",
    "Age",
    "Fare"
]

X = df[features]

y = df["Survived"]

# ---------------------------------
# NORMALIZATION
# ---------------------------------

scaler = MinMaxScaler()

X = scaler.fit_transform(X)

# ---------------------------------
# TRAIN TEST SPLIT
# ---------------------------------

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------
# TRAIN MODEL
# ---------------------------------

@st.cache_resource
def train_model():

    model = MLPClassifier(
        hidden_layer_sizes=(8,4),
        activation='relu',
        max_iter=500,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    return model

model = train_model()

# ---------------------------------
# EVALUATION
# ---------------------------------

y_pred = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

cm = confusion_matrix(
    y_test,
    y_pred
)

# ---------------------------------
# HEADER
# ---------------------------------

st.markdown(
"""
<div class='title'>
🚢 Titanic Survival Prediction System
</div>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<div class='subtitle'>
Deep Learning Based Passenger Survival Prediction
</div>
""",
unsafe_allow_html=True
)

c1,c2,c3=st.columns([2,1,2])

with c2:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/2784/2784445.png",
        width=100
    )

# ---------------------------------
# DESCRIPTION
# ---------------------------------

st.markdown(
"""
<div class='card'>

<h3>Project Description</h3>

<p>
Predict survival using ANN.
</p>

<p>
Model includes:
</p>

<p>✅ ANN</p>

<p>✅ Accuracy Evaluation</p>

<p>✅ Confusion Matrix</p>

<p>✅ Streamlit Deployment</p>

</div>
""",
unsafe_allow_html=True
)

# ---------------------------------
# MODEL PERFORMANCE
# ---------------------------------

st.markdown(
"""
<div class='card'>
<h3>Model Performance</h3>
</div>
""",
unsafe_allow_html=True
)

a,b=st.columns(2)

with a:

    st.metric(
        "Accuracy",
        f"{accuracy*100:.2f}%"
    )

with b:

    st.metric(
        "Total Test Samples",
        len(y_test)
    )

# ---------------------------------
# CONFUSION MATRIX
# ---------------------------------

st.subheader(
    "Confusion Matrix"
)

fig,ax=plt.subplots()

ax.imshow(cm)

for i in range(2):
    for j in range(2):

        ax.text(
            j,
            i,
            cm[i,j],
            ha='center'
        )

ax.set_xlabel(
    "Predicted"
)

ax.set_ylabel(
    "Actual"
)

st.pyplot(fig)

# ---------------------------------
# INPUT
# ---------------------------------

st.markdown(
"""
<div class='card'>
<h3>Passenger Input Form</h3>
</div>
""",
unsafe_allow_html=True
)

c1,c2,c3=st.columns(3)

with c1:

    pclass=st.selectbox(
        "Passenger Class",
        [1,2,3]
    )

with c2:

    age=st.slider(
        "Age",
        1,
        80,
        24
    )

with c3:

    fare=st.number_input(
        "Fare",
        0.0,
        600.0,
        50.0
    )

# ---------------------------------
# PREDICT
# ---------------------------------

if st.button(
    "Predict Survival"
):

    user=np.array([
        [pclass,age,fare]
    ])

    user=scaler.transform(
        user
    )

    prob=model.predict_proba(
        user
    )[0][1]

    non_prob=1-prob

    if prob>0.5:

        result="Survived"

        st.success(
            "Passenger likely survives"
        )

    else:

        result="Not Survived"

        st.error(
            "Passenger likely may not survive"
        )

    confidence=max(
        prob,
        non_prob
    )*100

    x,y,z=st.columns(3)

    with x:

        st.metric(
            "Prediction",
            result
        )

    with y:

        st.metric(
            "Survival Probability",
            f"{prob*100:.2f}%"
        )

    with z:

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

# ---------------------------------
# PIE CHART
# ---------------------------------

    st.subheader(
        "Probability Visualization"
    )

    p1,p2,p3=st.columns([1,2,1])

    with p2:

        fig,ax=plt.subplots(
            figsize=(3,3)
        )

        ax.pie(
            [prob,non_prob],
            labels=[
                "Survival",
                "Non Survival"
            ],
            autopct="%1.1f%%"
        )

        st.pyplot(
            fig
        )

st.markdown("---")

st.caption(
"Built using ANN + Streamlit"
)