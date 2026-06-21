import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score


BASE_DIR = Path(__file__).resolve().parent


def prediction_confidence(model, sample):
    probabilities = model.predict_proba(sample)[0]
    prediction = int(model.classes_[np.argmax(probabilities)])
    confidence = float(np.max(probabilities))
    return prediction, confidence


def label_prediction(prediction):
    return "Heart Disease" if prediction == 1 else "No Heart Disease"


def minmax_to_minus_one_one(value, min_value, max_value):
    return 2 * ((value - min_value) / (max_value - min_value)) - 1


def patient_from_webapp_input(values, columns):
    """Map raw UI values to the encoded feature scale used by the CSV files."""
    encoded = {
        "age": minmax_to_minus_one_one(values["age"], 29, 77),
        "trestbps": minmax_to_minus_one_one(values["trestbps"], 94, 200),
        "chol": minmax_to_minus_one_one(values["chol"], 126, 564),
        "thalach": minmax_to_minus_one_one(values["thalach"], 71, 202),
        "oldpeak": minmax_to_minus_one_one(values["oldpeak"], 0, 6.2),
        "sex": values["sex"],
        "cp": (values["cp"] - 1) / 3,
        "fbs": values["fbs"],
        "restecg": values["restecg"] / 2,
        "exang": values["exang"],
        "slope": (values["slope"] - 1) / 2,
        "ca": values["ca"] / 3,
        "thal": {3: 0, 6: 0.5, 7: 1}[values["thal"]],
    }
    return pd.DataFrame([encoded], columns=columns)


@st.cache_resource(show_spinner=False)
def load_trained_models():
    """Load pre-trained models and evaluation data from pickle file."""
    model_path = BASE_DIR / "models.pkl"
    with open(model_path, "rb") as f:
        payload = pickle.load(f)
    return (
        payload["models"],
        payload["feature_columns"],
        payload["x_val"],
        payload["y_val"],
        payload["x_test"],
        payload["y_test"],
    )


def evaluate_models(models, x_val, y_val, x_test, y_test, new_patient):
    results = []

    for model_name, model in models.items():
        val_predictions = model.predict(x_val)
        test_predictions = model.predict(x_test)
        patient_prediction, patient_confidence = prediction_confidence(
            model,
            new_patient,
        )

        results.append(
            {
                "model": model_name,
                "val_accuracy": accuracy_score(y_val, val_predictions),
                "test_accuracy": accuracy_score(y_test, test_predictions),
                "patient_prediction": label_prediction(patient_prediction),
                "patient_confidence": patient_confidence,
            }
        )

    return pd.DataFrame(results).sort_values(
        by=["val_accuracy", "patient_confidence"],
        ascending=False,
    )


EXAMPLES = {
    "Example 1 (No Heart Disease)": {
        "age": 58,
        "sex": 1,
        "cp": 2,
        "trestbps": 130,
        "chol": 250,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 1,
        "ca": 0,
        "thal": 3,
    },
    "Example 2 (Higher Risk)": {
        "age": 67,
        "sex": 1,
        "cp": 4,
        "trestbps": 160,
        "chol": 286,
        "fbs": 0,
        "restecg": 0,
        "thalach": 108,
        "exang": 1,
        "oldpeak": 1.5,
        "slope": 2,
        "ca": 3,
        "thal": 7,
    },
    "Example 3 (Lower Risk)": {
        "age": 45,
        "sex": 0,
        "cp": 1,
        "trestbps": 112,
        "chol": 160,
        "fbs": 0,
        "restecg": 1,
        "thalach": 185,
        "exang": 0,
        "oldpeak": 0.2,
        "slope": 1,
        "ca": 0,
        "thal": 3,
    },
}





def render_prediction_chart(results_df):
    labels = results_df["model"].tolist()
    confidences = results_df["patient_confidence"].tolist()
    predictions = results_df["patient_prediction"].tolist()
    colors = [
        "#bd3c4e" if prediction == "Heart Disease" else "#3c7f37"
        for prediction in predictions
    ]

    fig, ax = plt.subplots(figsize=(12, 5.8))
    bars = ax.bar(labels, confidences, color=colors, edgecolor="#222", linewidth=1.2)

    ax.set_title("Model Predictions", loc="left", fontsize=18, pad=12)
    ax.set_ylabel("Prediction Confidence", fontsize=12)
    ax.set_xlabel("Model", fontsize=12, labelpad=18)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="y", color="#d8d8d8", linestyle="-", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, confidence, prediction in zip(bars, confidences, predictions):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            confidence + 0.035,
            f"{confidence:.0%}",
            ha="center",
            va="bottom",
            fontsize=11,
            color="#111",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(confidence * 0.52, confidence - 0.06),
            prediction,
            ha="center",
            va="center",
            rotation=270,
            fontsize=10,
            color="#ffffff",
            fontweight="bold",
        )

    plt.xticks(rotation=-32, ha="left")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def render_result_metrics(results_df):
    best_confidence = results_df.sort_values(
        by="patient_confidence",
        ascending=False,
    ).iloc[0]
    best_validation = results_df.sort_values(
        by="val_accuracy",
        ascending=False,
    ).iloc[0]
    disease_votes = int((results_df["patient_prediction"] == "Heart Disease").sum())
    no_disease_votes = int((results_df["patient_prediction"] == "No Heart Disease").sum())

def render_input_form():
    selected_example = st.selectbox(
        "Select Example Patient",
        list(EXAMPLES.keys()),
        index=0,
    )
    defaults = EXAMPLES[selected_example]

    with st.form("patient-form"):
        st.markdown("### Enter Patient Features")

        row_1 = st.columns(4)
        age = row_1[0].number_input("age (years)", 1, 120, defaults["age"])
        sex = row_1[1].selectbox(
            "sex (0=female, 1=male)",
            [0, 1],
            index=[0, 1].index(defaults["sex"]),
        )
        cp = row_1[2].selectbox(
            "cp (chest pain type 1..4)",
            [1, 2, 3, 4],
            index=[1, 2, 3, 4].index(defaults["cp"]),
        )
        trestbps = row_1[3].number_input(
            "trestbps (resting BP mmHg)",
            70,
            220,
            defaults["trestbps"],
        )

        row_2 = st.columns(4)
        chol = row_2[0].number_input(
            "chol (serum cholesterol mg/dl)",
            100,
            650,
            defaults["chol"],
        )
        fbs = row_2[1].selectbox(
            "fbs (>120 mg/dl? 1/0)",
            [0, 1],
            index=[0, 1].index(defaults["fbs"]),
        )
        restecg = row_2[2].selectbox(
            "restecg (0..2)",
            [0, 1, 2],
            index=[0, 1, 2].index(defaults["restecg"]),
        )
        thalach = row_2[3].number_input(
            "thalach (max heart rate)",
            60,
            230,
            defaults["thalach"],
        )

        row_3 = st.columns(4)
        exang = row_3[0].selectbox(
            "exang (exercise angina 1/0)",
            [0, 1],
            index=[0, 1].index(defaults["exang"]),
        )
        oldpeak = row_3[1].number_input(
            "oldpeak (ST depression)",
            0.0,
            7.0,
            float(defaults["oldpeak"]),
            step=0.1,
        )
        slope = row_3[2].selectbox(
            "slope (1..3)",
            [1, 2, 3],
            index=[1, 2, 3].index(defaults["slope"]),
        )
        ca = row_3[3].selectbox(
            "ca (major vessels 0..3)",
            [0, 1, 2, 3],
            index=[0, 1, 2, 3].index(defaults["ca"]),
        )

        thal = st.selectbox(
            "thal (3=normal, 6=fixed, 7=reversible)",
            [3, 6, 7],
            index=[3, 6, 7].index(defaults["thal"]),
        )

        submitted = st.form_submit_button("Predict", width="stretch")

    values = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }
    return values, submitted


def render_page():
    st.set_page_config(
        page_title="Heart Disease Prediction Demo",
        page_icon="heart",
        layout="wide",
    )

    st.markdown(
        """
        <style>
            .stApp {
                background: #ffffff;
            }
            .main-title {
                color: #ef3124;
                font-size: 64px;
                font-weight: 800;
                line-height: 1;
                margin: 0 0 24px;
            }
            section[data-testid="stSidebar"] {
                display: none;
            }
            div[data-testid="stForm"] {
                background: #242424;
                border: 1px solid #3f3f3f;
                border-radius: 4px;
                padding: 18px 18px 8px;
            }
            div[data-testid="stForm"] label,
            div[data-testid="stForm"] p,
            div[data-testid="stForm"] span {
                color: #efefef;
                font-weight: 650;
            }
            div[data-testid="stForm"] input,
            div[data-testid="stForm"] div[data-baseweb="select"] > div {
                background: #454545;
                color: #ffffff;
                border-color: #454545;
            }
            .prediction-panel {
                border: 6px solid #222;
                border-radius: 2px;
                padding: 4px 10px 12px;
                background: #ffffff;
            }
            .panel-chip {
                display: inline-block;
                background: #333333;
                color: #f4f4f4;
                padding: 9px 16px;
                border-radius: 0 0 6px 0;
                font-size: 22px;
                font-weight: 800;
                margin: -4px 0 4px -10px;
            }
            div[data-testid="stMetric"] {
                background: #f7f7f7;
                border: 1px solid #dedede;
                border-radius: 6px;
                padding: 10px 12px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    models, feature_columns, x_val, y_val, x_test, y_test = load_trained_models()

    st.markdown('<h1 class="main-title">Web App Demo</h1>', unsafe_allow_html=True)

    left_col, right_col = st.columns([1.05, 1.15], gap="large")

    with left_col:
        patient_values, predict_clicked = render_input_form()

    with right_col:
        st.markdown('<div class="prediction-panel">', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-chip">Model Predictions Overview</div>',
            unsafe_allow_html=True,
        )

        if "patient_values" not in st.session_state:
            st.session_state["patient_values"] = EXAMPLES["Example 1 (No Heart Disease)"]
        if predict_clicked:
            st.session_state["patient_values"] = patient_values

        new_patient = patient_from_webapp_input(
            st.session_state["patient_values"],
            feature_columns,
        )
        results_df = evaluate_models(models, x_val, y_val, x_test, y_test, new_patient)

        render_prediction_chart(results_df)
        render_result_metrics(results_df)

       
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    render_page()
