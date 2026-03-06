import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from evaluation.realtime_accuracy import realtime_accuracy


# 📁 Where FAISS index will be stored
INDEX_PATH = "embeddings/faiss_index"


def build_rag_pipeline(chunks):

    # 🔐 Use environment variable (safer)
    embeddings = JinaEmbeddings(
        jina_api_key=os.getenv("JINA_API_KEY"),
        model_name="jina-embeddings-v3",
        request_batch_size=16   
    )

    # =========================================================
    # ✅ LOAD or CREATE VECTOR STORE (Fixes rate limit problem)
    # =========================================================
    if os.path.exists(INDEX_PATH):

        vectorstore = FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    else:

        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Save index so embeddings are not recreated
        vectorstore.save_local(INDEX_PATH)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # =========================================================
    # 🤖 Groq LLM
    # =========================================================
    llm = ChatOpenAI(
        model_name="llama-3.1-8b-instant",
        openai_api_key=os.getenv("GROQ_API_KEY"),
        openai_api_base="https://api.groq.com/openai/v1",
        temperature=0
    )

    # =========================================================
    # 🌐 Bilingual Legal Prompt
    # =========================================================
    template = """
You are a professional legal assistant for Bangladesh Labour Law.

Use ONLY the provided context to answer the question.

STRICT LANGUAGE RULE:

- If the user asks in English → Answer ONLY in English.
- If the user asks in Bangla → Answer ONLY in Bangla.
- Do NOT mix languages.

If answer is not found in context, say:

Bangla:
"তথ্যটি প্রদত্ত ডকুমেন্টে পাওয়া যায়নি।"

English:
"The information was not found in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

    QA_PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,   
        chain_type_kwargs={"prompt": QA_PROMPT}
    )

    return qa_chain