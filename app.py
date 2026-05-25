import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

with open("style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# ---------------------------
# DATA
# ---------------------------

df=pd.read_csv("Titanic-Dataset.csv")

df["Age"]=df["Age"].fillna(
    df["Age"].mean()
)

df["Embarked"]=df["Embarked"].fillna(
    df["Embarked"].mode()[0]
)

le=LabelEncoder()

df["Sex"]=le.fit_transform(
    df["Sex"]
)

features=[
    "Pclass",
    "Sex",
    "Age",
    "Fare"
]

X=df[features]

y=df["Survived"]

X_train,X_test,y_train,y_test=\
train_test_split(
X,
y,
test_size=0.2,
random_state=42
)

scaler=MinMaxScaler()

X_train=scaler.fit_transform(
    X_train
)

X_test=scaler.transform(
    X_test
)

# ---------------------------
# MODEL
# ---------------------------

@st.cache_resource
def train_model():

    model=MLPClassifier(

        hidden_layer_sizes=(32,16),

        activation='relu',

        solver='adam',

        max_iter=1500,

        early_stopping=True,

        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    return model

model=train_model()

pred=model.predict(X_test)

acc=accuracy_score(
    y_test,
    pred
)

cm=confusion_matrix(
    y_test,
    pred
)

# ---------------------------
# HEADER
# ---------------------------

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

# ---------------------------
# PERFORMANCE
# ---------------------------

a,b=st.columns(2)

with a:
    st.metric(
        "Accuracy",
        f"{acc*100:.2f}%"
    )

with b:
    st.metric(
        "Testing Samples",
        len(y_test)
    )

# ---------------------------
# MATRIX
# ---------------------------

st.subheader(
"Confusion Matrix"
)

x,y,z=st.columns([1,2,1])

with y:

    fig,ax=plt.subplots(
        figsize=(2.5,2.5)
    )

    ax.imshow(cm)

    for i in range(2):
        for j in range(2):

            ax.text(
                j,
                i,
                cm[i,j],
                ha="center"
            )

    st.pyplot(
        fig,
        use_container_width=False
    )

# ---------------------------
# INPUT
# ---------------------------

st.subheader(
"Passenger Input Form"
)

c1,c2=st.columns(2)

with c1:

    pclass=st.selectbox(
        "Passenger Class",
        [1,2,3]
    )

    sex=st.selectbox(
        "Gender",
        ["Male","Female"]
    )

with c2:

    age=st.slider(
        "Age",
        1,
        80,
        25
    )

    fare=st.number_input(
        "Fare",
        0,
        600,
        50
    )

sex=1 if sex=="Male" else 0

# ---------------------------
# PREDICT
# ---------------------------

if st.button(
"Predict Survival"
):

    user=np.array([[
        pclass,
        sex,
        age,
        fare
    ]])

    user=scaler.transform(
        user
    )

    prob=model.predict_proba(
        user
    )[0][1]

    non=1-prob

    result=(
        "Survived"
        if prob>0.5
        else
        "Not Survived"
    )

    a,b,c=st.columns(3)

    with a:

        st.metric(
            "Prediction",
            result
        )

    with b:

        st.metric(
            "Probability",
            f"{prob*100:.2f}%"
        )

    with c:

        st.metric(
            "Confidence",
            f"{max(prob,non)*100:.2f}%"
        )

    p1,p2,p3=st.columns([1,2,1])

    with p2:

        fig,ax=plt.subplots(
            figsize=(3,3)
        )

        ax.pie(
            [prob,non],
            labels=[
                "Survival",
                "Non-Survival"
            ],
            autopct="%1.1f%%"
        )

        st.pyplot(fig)