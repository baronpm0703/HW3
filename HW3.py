import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import FancyBboxPatch
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


MODEL_ORDER = [
    "Decision Tree",
    "K-NN",
    "Naive Bayes",
    "Random Forest",
    "AdaBoost",
    "Gradient Boosting",
    "XGBoost",
    "Ensemble (Soft Voting)",
]

METRIC_CLASSIFIER_ORDER = [
    "Decision Tree",
    "AdaBoost",
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
]


def render_prediction_chart(results_df):
    result_models = set(results_df["model"])
    ordered_results = (
        results_df.set_index("model")
        .reindex([model for model in MODEL_ORDER if model in result_models])
        .reset_index()
    )
    labels = ordered_results["model"].tolist()
    confidences = ordered_results["patient_confidence"].tolist()
    predictions = ordered_results["patient_prediction"].tolist()
    x_positions = np.arange(len(labels))
    colors = [
        "#c73251" if prediction == "Heart Disease" else "#2c7d31"
        for prediction in predictions
    ]

    fig = plt.figure(figsize=(12.4, 8.4), facecolor="#ffffff")
    panel = FancyBboxPatch(
        (0.01, 0.035),
        0.98,
        0.93,
        boxstyle="round,pad=0.0,rounding_size=0.004",
        transform=fig.transFigure,
        facecolor="#ffffff",
        edgecolor="#222222",
        linewidth=6,
        zorder=0,
    )
    chip = FancyBboxPatch(
        (0.024, 0.895),
        0.48,
        0.065,
        boxstyle="round,pad=0.012,rounding_size=0.011",
        transform=fig.transFigure,
        facecolor="#303030",
        edgecolor="#4a4a4a",
        linewidth=1,
        zorder=2,
    )
    fig.patches.extend([panel, chip])

    icon_ax = fig.add_axes([0.036, 0.912, 0.026, 0.034], zorder=3)
    icon_ax.plot(
        [0.12, 0.12, 0.9],
        [0.88, 0.12, 0.12],
        color="#bdbdbd",
        linewidth=1.6,
    )
    icon_ax.plot(
        [0.22, 0.43, 0.62, 0.82],
        [0.26, 0.5, 0.36, 0.72],
        color="#bdbdbd",
        linewidth=1.5,
    )
    icon_ax.scatter(
        [0.22, 0.43, 0.62, 0.82],
        [0.26, 0.5, 0.36, 0.72],
        s=9,
        color="#bdbdbd",
    )
    icon_ax.set_axis_off()

    fig.text(
        0.072,
        0.927,
        "Model Predictions Overview",
        color="#f2f2f2",
        fontsize=24,
        fontweight="bold",
        va="center",
        ha="left",
        zorder=3,
    )

    ax = fig.add_axes([0.12, 0.30, 0.74, 0.48], zorder=1)
    bars = ax.bar(
        x_positions,
        confidences,
        width=0.66,
        color=colors,
        edgecolor="#222222",
        linewidth=1.0,
    )

    ax.set_title("Model Predictions", loc="left", fontsize=25, pad=18)
    ax.set_ylabel("Prediction Confidence", fontsize=20, labelpad=20)
    ax.set_xlabel("Model", fontsize=19, labelpad=20)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"], fontsize=16)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#222222")
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.set_xticks(x_positions, labels, rotation=-32, ha="left", fontsize=16)

    for bar, confidence, prediction in zip(bars, confidences, predictions):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            confidence + 0.022,
            f"{confidence:.0%}",
            ha="center",
            va="bottom",
            fontsize=17,
            color="#111",
        )
        label = f"{'+' if prediction == 'Heart Disease' else '✓'}  {prediction}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(0.08, min(confidence * 0.55, confidence - 0.09)),
            label,
            ha="center",
            va="center",
            rotation=270,
            fontsize=16,
            color="#ffffff",
            fontweight="bold",
        )

    st.pyplot(fig, clear_figure=True, width="stretch")


def render_result_metrics(models):
    classifier_names = [
        models[model_name].__class__.__name__
        for model_name in METRIC_CLASSIFIER_ORDER
        if model_name in models
    ]
    st.markdown(
        f"""
        <div class="classifier-list">
            {"<br>".join(classifier_names)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_input_form():
    def render_field_label(container, label, compact=False):
        compact_class = " compact-field-label" if compact else ""
        container.markdown(
            f'<div class="field-label{compact_class}">{label}</div>',
            unsafe_allow_html=True,
        )

    example_names = list(EXAMPLES.keys())
    selected_example = st.session_state.get(
        "selected_example_patient",
        example_names[0],
    )
    defaults = EXAMPLES[selected_example]

    with st.form("patient-form"):
        st.markdown(
            """
            <div class="patient-form-header">
                <span>&#9997;</span>
                <span>Enter Patient Features</span>
                <span class="patient-form-caret">&#9660;</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        row_1 = st.columns(4)
        render_field_label(row_1[0], "age (years)")
        age = row_1[0].number_input(
            "age (years)",
            1,
            120,
            defaults["age"],
            label_visibility="collapsed",
        )
        render_field_label(row_1[1], "sex (0=female, 1=male)")
        sex = row_1[1].selectbox(
            "sex (0=female, 1=male)",
            [0, 1],
            index=[0, 1].index(defaults["sex"]),
            label_visibility="collapsed",
        )
        render_field_label(row_1[2], "cp (chest pain type 1..4)")
        cp = row_1[2].selectbox(
            "cp (chest pain type 1..4)",
            [1, 2, 3, 4],
            index=[1, 2, 3, 4].index(defaults["cp"]),
            label_visibility="collapsed",
        )
        render_field_label(row_1[3], "trestbps (resting BP mmHg)")
        trestbps = row_1[3].number_input(
            "trestbps (resting BP mmHg)",
            70,
            220,
            defaults["trestbps"],
            label_visibility="collapsed",
        )

        row_2 = st.columns(4)
        render_field_label(row_2[0], "chol (serum cholesterol mg/dl)")
        chol = row_2[0].number_input(
            "chol (serum cholesterol mg/dl)",
            100,
            650,
            defaults["chol"],
            label_visibility="collapsed",
        )
        render_field_label(row_2[1], "fbs (>120 mg/dl? 1/0)")
        fbs = row_2[1].selectbox(
            "fbs (>120 mg/dl? 1/0)",
            [0, 1],
            index=[0, 1].index(defaults["fbs"]),
            label_visibility="collapsed",
        )
        render_field_label(row_2[2], "restecg (0..2)")
        restecg = row_2[2].selectbox(
            "restecg (0..2)",
            [0, 1, 2],
            index=[0, 1, 2].index(defaults["restecg"]),
            label_visibility="collapsed",
        )
        render_field_label(row_2[3], "thalach (max heart rate)")
        thalach = row_2[3].number_input(
            "thalach (max heart rate)",
            60,
            230,
            defaults["thalach"],
            label_visibility="collapsed",
        )

        row_3 = st.columns(4)
        render_field_label(row_3[0], "exang (exercise angina 1/0)")
        exang = row_3[0].selectbox(
            "exang (exercise angina 1/0)",
            [0, 1],
            index=[0, 1].index(defaults["exang"]),
            label_visibility="collapsed",
        )
        render_field_label(row_3[1], "oldpeak (ST depression)")
        oldpeak = row_3[1].number_input(
            "oldpeak (ST depression)",
            0.0,
            7.0,
            float(defaults["oldpeak"]),
            step=0.1,
            label_visibility="collapsed",
        )
        render_field_label(row_3[2], "slope (1..3)")
        slope = row_3[2].selectbox(
            "slope (1..3)",
            [1, 2, 3],
            index=[1, 2, 3].index(defaults["slope"]),
            label_visibility="collapsed",
        )
        render_field_label(row_3[3], "ca (major vessels 0..3)")
        ca = row_3[3].selectbox(
            "ca (major vessels 0..3)",
            [0, 1, 2, 3],
            index=[0, 1, 2, 3].index(defaults["ca"]),
            label_visibility="collapsed",
        )

        thal_row = st.columns(1)
        render_field_label(
            thal_row[0],
            "thal (3=normal, 6=fixed, 7=reversible)",
            compact=True,
        )
        thal = thal_row[0].selectbox(
            "thal (3=normal, 6=fixed, 7=reversible)",
            [3, 6, 7],
            index=[3, 6, 7].index(defaults["thal"]),
            label_visibility="collapsed",
        )

        bottom_row = st.columns([0.95, 1.05])
        render_field_label(bottom_row[0], "Select Example Patient", compact=True)
        selected_example_input = bottom_row[0].selectbox(
            "Select Example Patient",
            example_names,
            index=example_names.index(selected_example),
            label_visibility="collapsed",
        )
        render_field_label(bottom_row[1], "&nbsp;", compact=True)
        submitted = bottom_row[1].form_submit_button("🔍 Predict", width="stretch")

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
    if submitted and selected_example_input != selected_example:
        st.session_state["selected_example_patient"] = selected_example_input
        values = EXAMPLES[selected_example_input]

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
            [data-testid="stForm"] {
                background: #242424;
                border: 1px solid #171717;
                border-radius: 0;
                padding: 0 6px 8px;
                overflow: hidden;
            }
            .patient-form-header {
                align-items: center;
                background: #242424;
                color: #eeeeee;
                display: flex;
                font-size: clamp(22px, 1.4vw, 26px);
                font-weight: 700;
                gap: 7px;
                line-height: 1.1;
                margin: 0 -6px 0;
                min-height: 58px;
                padding: 0 8px;
                position: relative;
            }
            .patient-form-caret {
                margin-left: auto;
            }
            [data-testid="stForm"] div[data-testid="stHorizontalBlock"] {
                border: 1px solid #3a3a3a;
                border-radius: 10px;
                margin: 0 0 22px;
                padding: 16px 18px 16px;
            }
            [data-testid="stForm"] div[data-testid="stElementContainer"] {
                margin-bottom: 0;
            }
            .field-label {
                color: #eeeeee;
                display: block;
                font-size: clamp(15px, 1.35vw, 26px);
                font-weight: 800;
                letter-spacing: 0;
                line-height: 1.28;
                margin: 0 0 12px;
                min-height: clamp(58px, 6.1vw, 112px);
            }
            .compact-field-label {
                min-height: 0;
            }
            [data-testid="stForm"] div[data-testid="stSelectbox"],
            [data-testid="stForm"] div[data-testid="stNumberInput"] {
                margin-bottom: 0;
            }
            [data-testid="stForm"] [data-testid="stNumberInputContainer"] {
                border: 1px solid #f2f2f2;
                border-radius: 10px;
                overflow: hidden;
            }
            [data-testid="stForm"] input,
            [data-testid="stForm"] div[data-baseweb="select"] > div {
                background: #454545;
                border: 0;
                border-radius: 10px;
                color: #ffffff;
                min-height: clamp(42px, 3.3vw, 60px);
            }
            [data-testid="stForm"] [data-testid="stNumberInputContainer"] input {
                border-radius: 0;
            }
            [data-testid="stForm"] input {
                font-size: clamp(18px, 1.45vw, 28px);
                padding-left: 16px;
            }
            [data-testid="stForm"] div[data-baseweb="select"] span,
            [data-testid="stForm"] div[data-baseweb="select"] svg {
                color: #ffffff;
                fill: #ffffff;
                font-size: clamp(18px, 1.45vw, 28px);
            }
            div[data-testid="stFormSubmitButton"] button {
                background: #5d5d5d;
                border: 0;
                border-radius: 10px;
                color: #ffffff;
                font-size: clamp(22px, 1.7vw, 31px);
                font-weight: 800;
                min-height: clamp(54px, 3.8vw, 68px);
                margin-top: 0;
            }
            div[data-testid="stFormSubmitButton"] button p,
            div[data-testid="stFormSubmitButton"] button span {
                color: #ffffff;
                font-size: clamp(22px, 1.7vw, 31px);
                font-weight: 800;
                line-height: 1;
            }
            div[data-testid="stFormSubmitButton"] button:hover,
            div[data-testid="stFormSubmitButton"] button:focus {
                background: #656565;
                border: 0;
                color: #ffffff;
            }
            .classifier-list {
                color: #000000;
                font-size: 38px;
                font-weight: 500;
                line-height: 1.16;
                margin: 18px 0 0 44px;
                letter-spacing: 0;
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
        render_result_metrics(models)


if __name__ == "__main__":
    render_page()
