import json
import pandas as pd
import os
import numpy as np
from datetime import datetime

from langchain_community.embeddings import JinaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, confusion_matrix

from evaluation.bertscore_eval import bertscore_similarity


# ===============================
# Cosine Similarity Function
# ===============================
def semantic_similarity(answer_true, answer_pred, embeddings):

    emb1 = embeddings.embed_query(answer_true)
    emb2 = embeddings.embed_query(answer_pred)

    score = cosine_similarity([emb1], [emb2])[0][0]

    return score


# ===============================
# Main Evaluation Function
# ===============================
def evaluate_rag(qa_chain):

    # Load dataset
    with open("evaluation/test_set.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Initialize embeddings ONCE
    embeddings = JinaEmbeddings(
        jina_api_key=os.getenv("JINA_API_KEY"),
        model_name="jina-embeddings-v3"
    )

    results = []
    cosine_scores = []
    bert_scores = []

    y_true = []
    y_pred = []

    for item in test_data:

        question = item["question"]
        true_answer = item["answer"]

        # -------------------------------
        # Get RAG prediction
        # -------------------------------
        response = qa_chain({"query": question})
        predicted_answer = response["result"]

        # -------------------------------
        # Cosine Similarity
        # -------------------------------
        cos_score = semantic_similarity(
            true_answer,
            predicted_answer,
            embeddings
        )

        # -------------------------------
        # BERTScore
        # -------------------------------
        bert_score = bertscore_similarity(
            true_answer,
            predicted_answer
        )

        # -------------------------------
        # Hybrid Score
        # -------------------------------
        final_score = (cos_score + bert_score) / 2

        cosine_scores.append(cos_score)
        bert_scores.append(bert_score)

        # Classification threshold
        if final_score >= 0.75:
            y_pred.append(1)
        else:
            y_pred.append(0)

        # Ground truth = correct answer expected
        y_true.append(1)

        results.append({
            "Question": question,
            "Cosine": round(cos_score, 3),
            "BERTScore": round(bert_score, 3),
            "Final Score": round(final_score, 3)
        })

    # ===============================
    # Metrics
    # ===============================
    overall_accuracy = np.mean(
        [r["Final Score"] for r in results]
    )

    avg_cosine = np.mean(cosine_scores)
    avg_bert = np.mean(bert_scores)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)

    df = pd.DataFrame(results)

    # ===============================
    # 💾 SAVE RESULTS
    # ===============================
    os.makedirs("evaluation/results", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df.to_csv(
        f"evaluation/results/eval_{timestamp}.csv",
        index=False
    )

    #df.to_excel(
       # f"evaluation/results/eval_{timestamp}.xlsx",
        #index=False
    #)

    df.to_json(
        f"evaluation/results/eval_{timestamp}.json",
        orient="records",
        force_ascii=False
    )

    # ===============================
    # Return Metrics
    # ===============================
    return (
        overall_accuracy,
        avg_cosine,
        avg_bert,
        precision,
        recall,
        cm,
        df
    )