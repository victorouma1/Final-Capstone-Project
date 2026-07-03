"""
AeroBERT Safety Report Classifier — Streamlit App
---------------------------------------------------
Deploys a fine-tuned NASA-AIML/MIKA_SafeAeroBERT sequence classification
model (trained in the accompanying notebook on ASRS aviation safety
narratives) behind a colorful, easy-to-use Streamlit UI.

Users can either:
  1. Type / paste a single narrative and get an instant prediction, or
  2. Upload a CSV(either already containing a "Narrative" column, or a
     raw ASRS export with "Assessments.1" / "Report 1" columns) and get batch predictions + a downloadable
     results file.

Run with:
    streamlit run app.py
"""

import io
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

st.set_page_config(
    page_title="AeroBERT Safety Report Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* ---------- Global background ---------- */
.stApp {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 45%, #2c5364 100%);
    color: #f5f7fa;
}

/* ---------- Headline banner ---------- */
.hero {
    padding: 2rem 2rem 1.6rem 2rem;
    border-radius: 20px;
    background: linear-gradient(120deg, #ff6a00 0%, #ee0979 50%, #7b2ff7 100%);
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    margin-bottom: 1.6rem;
}
.hero h1 {
    color: white;
    font-size: 2.4rem;
    margin-bottom: 0.2rem;
    text-shadow: 0 2px 8px rgba(0,0,0,0.25);
}
.hero p {
    color: #fdf1ff;
    font-size: 1.05rem;
    margin: 0;
}

/* ---------- Cards ---------- */
.metric-card {
    padding: 1.1rem 1.3rem;
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255,255,255,0.12);
    backdrop-filter: blur(6px);
}

/* ---------- Tabs ---------- */
button[data-baseweb="tab"] {
    font-size: 1.05rem;
    font-weight: 600;
}

/* ---------- Buttons ---------- */
div.stButton > button {
    background: linear-gradient(90deg, #ff6a00, #ee0979);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 1.4rem;
    font-weight: 700;
    font-size: 1rem;
    box-shadow: 0 6px 16px rgba(238,9,121,0.35);
    transition: transform 0.15s ease;
}
div.stButton > button:hover {
    transform: translateY(-2px) scale(1.02);
    box-shadow: 0 10px 22px rgba(238,9,121,0.45);
}

/* ---------- Dataframe / uploader tweaks ---------- */
[data-testid="stFileUploaderDropzone"] {
    border-radius: 14px;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #141e30 0%, #243b55 100%);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
        <h1> AeroBERT Safety Report Classifier</h1>
        <p>Fine-tuned SafeAeroBERT model for triaging aviation safety narratives
        into their <b>Primary Problem</b> category — instantly, in your browser.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Label mapping (from the training notebook's LabelEncoder classes_)
# NOTE: this mapping is only correct if your saved model was trained with the
# exact same LabelEncoder class order as the notebook. Edit in the sidebar
# if your model uses a different label order.
# ----------------------------------------------------------------------------
DEFAULT_LABEL_MAPPING = {
    "0": "ATC Equipment / Nav Facility / Buildings",
    "1": "Aircraft",
    "2": "Airport",
    "3": "Airspace Structure",
    "4": "Ambigous",
    "5": "Chart Or Publication",
    "6": "Company Policy",
    "7": "Environment - Non Weather Related",
    "8": "Equipment / Tooling",
    "9": "Human Factors",
    "10": "Incorrect / Not Installed / Unavailable Part",
    "11": "Logbook Entry",
    "12": "MEL",
    "13": "Manuals",
    "14": "Procedure",
    "15": "Software and Automation",
    "16": "Staffing",
    "17": "Weather",
}

# A splash of color per class so charts/badges look lively and consistent
PALETTE = px.colors.qualitative.Bold + px.colors.qualitative.Vivid


def color_for_label(label: str) -> str:
    labels_sorted = sorted(DEFAULT_LABEL_MAPPING.values())
    idx = labels_sorted.index(label) if label in labels_sorted else 0
    return PALETTE[idx % len(PALETTE)]

with st.sidebar:
    st.markdown("## Model Settings")
    model_path = st.text_input(
        "Model path or Hugging Face repo id",
        value="./saved_aerobert_model_2",
        help=(
            "Path to the folder saved with model.save_pretrained() / "
            "tokenizer.save_pretrained() in the notebook (e.g. "
            "'./saved_aerobert_model_2'), or a Hugging Face Hub repo id."
        ),
    )

    st.markdown("### Label mapping")
    st.caption(
        "Maps the model's numeric class ids to human-readable labels. "
        "Defaults to the mapping used in the training notebook — edit the "
        "JSON below if your model's LabelEncoder order differs."
    )
    label_json = st.text_area(
        "Label mapping (JSON: id → label)",
        value=json.dumps(DEFAULT_LABEL_MAPPING, indent=2),
        height=220,
    )
    try:
        LABEL_MAPPING = {int(k): v for k, v in json.loads(label_json).items()}
    except Exception:
        st.error("Invalid JSON in label mapping — using default mapping instead.")
        LABEL_MAPPING = {int(k): v for k, v in DEFAULT_LABEL_MAPPING.items()}

    max_length = st.slider("Max token length", 32, 512, 256, step=32)
    batch_size = st.slider("Batch size (CSV predictions)", 1, 32, 8)

    load_clicked = st.button("Load / Reload model", use_container_width=True)

    st.markdown("---")
    st.caption(
        "💡 This app expects a model fine-tuned like in the notebook: "
        "`NASA-AIML/MIKA_SafeAeroBERT` fine-tuned for sequence "
        "classification on ASRS 'Primary Problem' labels."
    )


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


if "model_bundle" not in st.session_state:
    st.session_state.model_bundle = None
    st.session_state.model_path_loaded = None

if load_clicked or (
    st.session_state.model_bundle is None and model_path
):
    with st.spinner(f"Loading model from `{model_path}` ..."):
        try:
            st.session_state.model_bundle = load_model(model_path)
            st.session_state.model_path_loaded = model_path
            st.success("Model loaded successfully!")
        except Exception as e:
            st.session_state.model_bundle = None
            st.error(f"Couldn't load model from '{model_path}':\n\n{e}")

model_ready = st.session_state.model_bundle is not None


def predict_texts(texts, tokenizer, model, device, max_length=256, batch_size=8):
    """Returns list of (predicted_label_id, confidence, full_prob_array)."""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = [str(t).lower().strip() for t in texts[i : i + batch_size]]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)
        for p, prob_row in zip(preds, probs):
            results.append((int(p), float(prob_row[p]), prob_row))
    return results


def preprocess_uploaded_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts either:
      - A CSV that already has a 'Narrative' column, or
      - A raw ASRS export like in the notebook, with 'Assessments.1' and
        'Report 1' columns where the real header lives in the first data row.
    Returns a DataFrame with a single 'Narrative' column.
    """
    if "Narrative" in df.columns:
        out = df[["Narrative"]].dropna().reset_index(drop=True)
        return out

    if "Assessments.1" in df.columns and "Report 1" in df.columns:
        d = df[["Assessments.1", "Report 1"]].copy()
        d.dropna(inplace=True)
        if d.empty:
            raise ValueError("No usable rows found after dropping missing values.")
        d.columns = d.iloc[0]
        d = d.drop(d.index[0]).reset_index(drop=True)
        if "Narrative" not in d.columns:
            raise ValueError(
                "Detected an ASRS-style export, but no 'Narrative' column "
                "was found after promoting the header row."
            )
        out = d[["Narrative"]].dropna().reset_index(drop=True)
        return out

    raise ValueError(
        "CSV must either contain a 'Narrative' column directly, or the raw "
        "ASRS export columns 'Assessments.1' and 'Report 1' (as used in the "
        "training notebook)."
    )


# ----------------------------------------------------------------------------
# Main tabs: single text vs. CSV upload
# ----------------------------------------------------------------------------
tab_text, tab_csv, tab_about = st.tabs(
    ["Single Narrative", "Upload CSV", "About"]
)

with tab_text:
    st.markdown("#### Paste an aviation safety narrative below")
    example = (
        "During the initial climb out of JFK International Airport, the flight "
        "crew noted a sudden, abnormal vibration accompanied by a rapid rise in "
        "Exhaust Gas Temperature (EGT) on the No. 2 (right) engine."
    )
    user_text = st.text_area(
        "Narrative text", value="", placeholder=example, height=160
    )
    col_a, col_b = st.columns([1, 5])
    with col_a:
        predict_clicked = st.button("Classify", key="predict_single")
    with col_b:
        if st.button("Use example narrative"):
            user_text = example
            st.session_state["_example_used"] = example

    if predict_clicked:
        text_to_use = user_text.strip() or st.session_state.get("_example_used", "")
        if not text_to_use:
            st.warning("Please enter some text first.")
        elif not model_ready:
            st.warning("Load a model in the sidebar first.")
        else:
            tokenizer, model, device = st.session_state.model_bundle
            (pred_id, confidence, prob_row), = predict_texts(
                [text_to_use], tokenizer, model, device, max_length, batch_size=1
            )
            pred_label = LABEL_MAPPING.get(pred_id, f"LABEL_{pred_id}")
            color = color_for_label(pred_label)

            st.markdown("### Result")
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div style="font-size:0.9rem; opacity:0.8;">Predicted Primary Problem</div>
                        <div style="font-size:1.6rem; font-weight:800; color:{color};">
                            {pred_label}
                        </div>
                        <div style="margin-top:0.4rem; font-size:1rem;">
                            Confidence: <b>{confidence*100:.1f}%</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c2:
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=confidence * 100,
                        number={"suffix": "%"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": color},
                            "bgcolor": "rgba(255,255,255,0.05)",
                        },
                        title={"text": "Model Confidence"},
                    )
                )
                fig.update_layout(
                    height=220,
                    margin=dict(t=40, b=0, l=20, r=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#f5f7fa",
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Full probability distribution")
            prob_df = pd.DataFrame(
                {
                    "Label": [
                        LABEL_MAPPING.get(i, f"LABEL_{i}")
                        for i in range(len(prob_row))
                    ],
                    "Probability": prob_row * 100,
                }
            ).sort_values("Probability", ascending=True)
            fig_bar = px.bar(
                prob_df,
                x="Probability",
                y="Label",
                orientation="h",
                color="Label",
                color_discrete_sequence=PALETTE,
                text=prob_df["Probability"].round(1).astype(str) + "%",
            )
            fig_bar.update_layout(
                showlegend=False,
                height=520,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#f5f7fa",
                xaxis_title="Probability (%)",
                yaxis_title="",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

# ---- TAB 2: CSV upload -------------------------------------------------
with tab_csv:
    st.markdown(
        "#### Upload a CSV — either one with a `Narrative` column, or a raw "
        "ASRS export with `Assessments.1` / `Report 1` "
        "columns."
    )
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            narratives_df = preprocess_uploaded_csv(raw_df)
            st.success(
                f"Loaded {len(narratives_df)} narrative(s) from '{uploaded_file.name}'."
            )
            with st.expander("Preview parsed narratives", expanded=False):
                st.dataframe(narratives_df.head(10), use_container_width=True)

            run_batch = st.button("Classify all narratives", key="predict_batch")

            if run_batch:
                if not model_ready:
                    st.warning("Load a model in the sidebar first.")
                else:
                    tokenizer, model, device = st.session_state.model_bundle
                    progress = st.progress(0.0, text="Classifying narratives...")
                    texts = narratives_df["Narrative"].tolist()
                    all_results = []
                    step = max(1, len(texts) // 20)
                    processed = 0
                    for i in range(0, len(texts), batch_size):
                        chunk = texts[i : i + batch_size]
                        all_results.extend(
                            predict_texts(
                                chunk, tokenizer, model, device, max_length, batch_size
                            )
                        )
                        processed += len(chunk)
                        progress.progress(
                            min(processed / len(texts), 1.0),
                            text=f"Classifying narratives... ({processed}/{len(texts)})",
                        )
                    progress.empty()

                    pred_ids = [r[0] for r in all_results]
                    confidences = [r[1] for r in all_results]
                    results_df = narratives_df.copy()
                    results_df["Predicted_Label"] = [
                        LABEL_MAPPING.get(pid, f"LABEL_{pid}") for pid in pred_ids
                    ]
                    results_df["Confidence (%)"] = [
                        round(c * 100, 2) for c in confidences
                    ]

                    st.markdown("### Results")
                    st.dataframe(results_df, use_container_width=True, height=380)

                    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download results as CSV",
                        data=csv_bytes,
                        file_name="aerobert_predictions.csv",
                        mime="text/csv",
                    )

                    st.markdown("### Class distribution")
                    dist_df = (
                        results_df["Predicted_Label"]
                        .value_counts()
                        .reset_index()
                    )
                    dist_df.columns = ["Label", "Count"]
                    fig_dist = px.bar(
                        dist_df,
                        x="Count",
                        y="Label",
                        orientation="h",
                        color="Label",
                        color_discrete_sequence=PALETTE,
                        text="Count",
                    )
                    fig_dist.update_layout(
                        showlegend=False,
                        height=520,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#f5f7fa",
                        yaxis={"categoryorder": "total ascending"},
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

                    avg_conf = np.mean(confidences) * 100
                    st.markdown(
                        f"""
                        <div class="metric-card" style="max-width:320px;">
                            <div style="font-size:0.9rem; opacity:0.8;">Average confidence</div>
                            <div style="font-size:1.8rem; font-weight:800;">{avg_conf:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        except Exception as e:
            st.error(f"Couldn't process this CSV: {e}")

with tab_about:
    st.markdown(
        """
        #### About this app
        This app serves a fine-tuned **SafeAeroBERT**
        (`NASA-AIML/MIKA_SafeAeroBERT`) sequence classification model,
        trained in the accompanying notebook to categorize aviation safety
        narratives (e.g. from the FAA's ASRS database) into a **Primary
        Problem** class such as *Human Factors*, *Weather*, *Aircraft*, etc.
        """
    )