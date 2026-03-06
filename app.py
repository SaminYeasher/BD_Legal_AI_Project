import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# 🔐 Load Environment Variables
# ===============================
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["JINA_API_KEY"] = os.getenv("JINA_API_KEY")

# ===============================
# 📊 Evaluation Imports
# ===============================
from evaluation.accuracy import evaluate_rag
from evaluation.realtime_accuracy import realtime_accuracy
from evaluation.bertscore_eval import bertscore_similarity

# ===============================
# 🧠 RAG Imports
# ===============================
from pdf_loader import load_and_split_pdf
from rag_pipeline import build_rag_pipeline


# ===============================
# 🖥️ Page Config
# ===============================
st.set_page_config(
    page_title="Smart AI Legal Assistant",
    page_icon="🇧🇩",
    layout="wide"
)

st.title("🇧🇩 Smart AI Legal Assistant")
st.caption("Bangladesh Labour Law • Bangla + English Q&A")


# ===============================
# ⚙️ Setup RAG Pipeline
# ===============================
@st.cache_resource
def setup():

    chunks_en = load_and_split_pdf("data/bd_labor_law.pdf")
    chunks_bn = load_and_split_pdf("data/bd_labor_law_Bangla.pdf")

    all_chunks = chunks_en + chunks_bn

    qa_chain = build_rag_pipeline(all_chunks)

    return qa_chain


qa_chain = setup()


# ===============================
# 📊 Sidebar Evaluation Panel
# ===============================
st.sidebar.title("📊 System Evaluation")

if st.sidebar.button("Run Full Evaluation (60 Questions)"):

    with st.spinner("Running full evaluation... Please wait..."):

        (
            overall,
            avg_cosine,
            avg_bert,
            precision,
            recall,
            cm,
            df
        ) = evaluate_rag(qa_chain)

    # ===============================
    # Metrics Display
    # ===============================
    st.sidebar.success(f"Overall Accuracy: {overall*100:.2f}%")
    st.sidebar.info(f"Cosine Similarity: {avg_cosine*100:.2f}%")
    st.sidebar.info(f"BERTScore F1: {avg_bert*100:.2f}%")
    st.sidebar.info(f"Precision: {precision*100:.2f}%")
    st.sidebar.info(f"Recall: {recall*100:.2f}%")

    # ===============================
    # Results Table
    # ===============================
    st.subheader("📋 Evaluation Results")
    st.dataframe(df)

    # ===============================
    # 📉 Confusion Matrix
    # ===============================
    st.subheader("📉 Confusion Matrix")

    fig_cm, ax_cm = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Incorrect", "Correct"],
        yticklabels=["Incorrect", "Correct"],
        ax=ax_cm
    )

    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")

    st.pyplot(fig_cm)

    # ===============================
    # 📊 Precision vs Recall Bar Chart
    # ===============================
    st.subheader("📊 Precision vs Recall")

    fig_pr, ax_pr = plt.subplots()

    metrics = ["Precision", "Recall"]
    values = [precision, recall]

    ax_pr.bar(metrics, values)
    ax_pr.set_ylim(0, 1)
    ax_pr.set_ylabel("Score")

    st.pyplot(fig_pr)

    # ===============================
    # 📈 Cosine vs BERT Comparison
    # ===============================
    st.subheader("📈 Cosine vs BERTScore Comparison")

    fig_cb, ax_cb = plt.subplots()

    ax_cb.scatter(
        df["Cosine"],
        df["BERTScore"]
    )

    ax_cb.set_xlabel("Cosine Similarity")
    ax_cb.set_ylabel("BERTScore")
    ax_cb.set_title("Cosine vs BERTScore")

    st.pyplot(fig_cb)


# ===============================
# 💬 Chat History State
# ===============================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None


# ===============================
# Show Previous Messages
# ===============================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ===============================
# ⌨️ Chat Input
# ===============================
prompt = st.chat_input(
    "Ask your legal question... / আপনার আইনি প্রশ্ন লিখুন..."
)

if prompt:

    # Save user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # ===============================
    # 🤖 Assistant Response
    # ===============================
    with st.chat_message("assistant"):

        with st.spinner(
            "Getting legal information... / আইনি তথ্য খোঁজা হচ্ছে..."
        ):

            result = qa_chain({"query": prompt})
            answer = result["result"]

            st.session_state.last_result = result
            st.session_state.last_answer = answer

            st.markdown(answer)

            # ===============================
            # 📊 Real-time Cosine Accuracy
            # ===============================
            cos_score = realtime_accuracy(result)

            st.info(
                f"📊 Cosine Accuracy: {cos_score*100:.2f}%"
            )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )


# ===============================
# 🧠 Optional BERTScore (Realtime)
# ===============================
if st.session_state.last_result is not None:

    st.divider()
    st.subheader("🧠 Advanced Answer Evaluation")

    if st.checkbox("Run BERTScore (Slow)", key="bert_checkbox"):

        with st.spinner("Calculating BERTScore..."):

            result = st.session_state.last_result
            answer = st.session_state.last_answer

            context_text = " ".join(
                [
                    doc.page_content
                    for doc in result["source_documents"]
                ]
            )

            bert_score = bertscore_similarity(
                context_text,
                answer
            )

            st.info(
                f"🧠 BERTScore: {bert_score*100:.2f}%"
            )

            # Hybrid score
            cos_score = realtime_accuracy(result)
            final_score = (cos_score + bert_score) / 2

            st.success(
                f"📊 Final Hybrid Accuracy: {final_score*100:.2f}%"
            )