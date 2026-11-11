import json
from pathlib import Path
import joblib
import streamlit as st
import pandas as pd
from lib.data import MODEL_DIR, ASPECTS

st.title("Aspect-Based Sentiment Analysis")
st.caption("Run predictions using the saved scikit-learn ABSA models")

@st.cache_resource
def load_model_artifacts():
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
    lr = joblib.load(MODEL_DIR / "logistic_regression.joblib")
    svm = joblib.load(MODEL_DIR / "linear_svm.joblib")
    labels_path = MODEL_DIR / "labels.json"
    labels = None
    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
    return vectorizer, lr, svm, labels

try:
    vectorizer, lr_model, svm_model, labels_json = load_model_artifacts()
except Exception as exc:
    st.error(
        f"Could not load the model artifacts from {MODEL_DIR}. "
        f"Check that the vectorizer, both classifiers, and labels.json are present. "
        f"Details: {exc}"
    )
    st.stop()

model_choice = st.selectbox(
    "Model",
    ["Logistic Regression", "Linear SVM"]
)
aspect = st.selectbox("Hotel aspect", ASPECTS)
review = st.text_area(
    "Review text",
    placeholder="Example: The room was clean, but the bed was uncomfortable...",
    height=160,
)

if st.button("Predict sentiment", type="primary", disabled=not review.strip()):
    # This must match the exact prefixing used during model training.
    model_text = f"aspect_{aspect} {review.strip()}"
    features = vectorizer.transform([model_text])

    model = lr_model if model_choice == "Logistic Regression" else svm_model
    prediction = model.predict(features)[0]

    st.subheader("Prediction")
    st.success(f"Predicted sentiment: **{str(prediction).title()}**")

    # Decision scores are useful for both estimators, but are not probabilities.
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(features)[0]
        score_df = pd.DataFrame({
            "Sentiment": model.classes_,
            "Model probability": scores,
        }).sort_values("Model probability", ascending=False)
        st.caption("Class probabilities from Logistic Regression")
        st.bar_chart(score_df.set_index("Sentiment"))
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(features)
        if getattr(scores, "ndim", 1) == 1:
            scores = scores.reshape(1, -1)
        classes = list(model.classes_)
        if scores.shape[1] == len(classes):
            score_df = pd.DataFrame({
                "Sentiment": classes,
                "Decision score": scores[0],
            }).sort_values("Decision score", ascending=False)
            st.caption(
                "Linear SVM decision scores (not probabilities; "
                "do not interpret them as confidence percentages)."
            )
            st.bar_chart(score_df.set_index("Sentiment"))

st.divider()
st.markdown("""
**How this page works**

The selected aspect is prepended to the review using the same format used
during training: `aspect_<aspect> <review text>`. The saved TF-IDF vectorizer
then transforms the text before the selected classifier predicts its label.
""")

with st.expander("Model classes and artifact information"):
    st.write("Logistic Regression classes:", list(lr_model.classes_))
    st.write("Linear SVM classes:", list(svm_model.classes_))
    st.write("Vectorizer vocabulary size:", len(vectorizer.vocabulary_))
    st.write("labels.json:", labels_json)
