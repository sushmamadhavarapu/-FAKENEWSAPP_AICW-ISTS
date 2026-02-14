import streamlit as st
import pickle

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detection App")
st.write("Enter news content to check whether it is REAL or FAKE.")

@st.cache_resource
def load_model():
    with open("NEW_LRmodel.pkl", "rb") as f:
        model = pickle.load(f)

    with open("NEW_VECTmodel.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


model, vectorizer = load_model()

news_text = st.text_area("Enter News Text:", height=200)

if st.button("Predict"):

    if news_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        transformed_input = vectorizer.transform([news_text])
        prediction = model.predict(transformed_input)

        if prediction[0] == "FAKE":
            st.error("🚨 This News is FAKE")
        else:
            st.success("✅ This News is REAL")
