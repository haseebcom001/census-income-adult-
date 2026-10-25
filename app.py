import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Adult Income Predictor", page_icon="💰")

@st.cache_resource
def load_model():
    return joblib.load(Path(__file__).parent / "best_income_model.pkl")

bundle = load_model()
model = bundle["model"]
label_encoder = bundle["label_encoder"]
feature_columns = bundle["feature_columns"]

EDU_NUM = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5,
    "10th": 6, "11th": 7, "12th": 8, "HS-grad": 9, "Some-college": 10,
    "Assoc-voc": 11, "Assoc-acdm": 12, "Bachelors": 13, "Masters": 14,
    "Prof-school": 15, "Doctorate": 16,
}
WORKCLASS = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
             "Local-gov", "State-gov", "Without-pay", "Never-worked"]
MARITAL = ["Married-civ-spouse", "Divorced", "Never-married", "Separated",
           "Widowed", "Married-spouse-absent", "Married-AF-spouse"]
OCCUPATION = ["Tech-support", "Craft-repair", "Other-service", "Sales",
              "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
              "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
              "Transport-moving", "Priv-house-serv", "Protective-serv",
              "Armed-Forces"]
RELATIONSHIP = ["Wife", "Own-child", "Husband", "Not-in-family",
                "Other-relative", "Unmarried"]
RACE = ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]
SEX = ["Male", "Female"]
COUNTRY = ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada",
           "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece",
           "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy",
           "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland",
           "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti",
           "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland",
           "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru",
           "Hong", "Holand-Netherlands"]

st.title("💰 Adult Income Predictor")
st.write("Predict whether a person earns **more than $50K per year** "
         "using a HistGradient Boosting model trained on the UCI Adult dataset.")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 17, 90, 38)
    workclass = st.selectbox("Work class", WORKCLASS)
    education = st.selectbox("Education", list(EDU_NUM.keys()), index=8)
    marital = st.selectbox("Marital status", MARITAL)
    occupation = st.selectbox("Occupation", OCCUPATION)
    relationship = st.selectbox("Relationship", RELATIONSHIP)

with col2:
    race = st.selectbox("Race", RACE)
    sex = st.selectbox("Sex", SEX)
    hours = st.slider("Hours per week", 1, 99, 40)
    capital_gain = st.number_input("Capital gain", 0, 99999, 0)
    capital_loss = st.number_input("Capital loss", 0, 4356, 0)
    country = st.selectbox("Native country", COUNTRY)

if st.button("Predict income", type="primary"):
    row = {
        "age": age,
        "workclass": workclass,
        "fnlwgt": 189778,  # dataset median; not a meaningful user input
        "education": education,
        "education-num": EDU_NUM[education],
        "marital-status": marital,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours,
        "native-country": country,
    }
    input_df = pd.DataFrame([row])[feature_columns]

    pred = model.predict(input_df)[0]
    label = label_encoder.inverse_transform([pred])[0]
    proba = model.predict_proba(input_df)[0]
    classes = list(label_encoder.classes_)
    p_high = proba[classes.index(">50K")]

    if label == ">50K":
        st.success(f"Predicted income: **{label}**")
    else:
        st.info(f"Predicted income: **{label}**")
    st.write(f"Probability of earning >50K: **{p_high:.1%}**")
    st.progress(float(p_high))
