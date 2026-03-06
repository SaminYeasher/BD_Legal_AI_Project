import os
from langchain_community.embeddings import JinaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity


def realtime_accuracy(result):
    """
    Calculates semantic similarity between generated answer
    and retrieved context in real time.
    """

    # Initialize embeddings INSIDE function
    embeddings = JinaEmbeddings(
        jina_api_key=os.getenv("JINA_API_KEY"),
        model_name="jina-embeddings-v3"
    )

    answer = result["result"]

    # Merge retrieved context
    context_text = " ".join(
        [doc.page_content for doc in result["source_documents"]]
    )

    # Generate embeddings
    emb_answer = embeddings.embed_query(answer)
    emb_context = embeddings.embed_query(context_text)

    # Cosine similarity
    score = cosine_similarity(
        [emb_answer], [emb_context]
    )[0][0]

    return round(score, 3)