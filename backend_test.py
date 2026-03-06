import os


# ==============================
# LangChain Imports
# ==============================

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings



from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate


# ==============================
# 1️⃣ Load API Key (Gemini only)
# ==============================

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file")

os.environ["GOOGLE_API_KEY"] = api_key


# ==============================
# 2️⃣ Load Legal PDF
# ==============================

print("1. Loading the Legal Document...")

loader = PyPDFLoader("data/bd_labor_law.pdf")
docs = loader.load()


# ==============================
# 3️⃣ Split Text into Chunks
# ==============================

print("2. Splitting text into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

splits = text_splitter.split_documents(docs)


# ==============================
# 4️⃣ Create Vector DB (Local Embeddings)
# ==============================

print("3. Creating Vector Database (Using HuggingFace Embeddings)...")

# ✅ Local embedding model (downloads once)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings
)

retriever = vectorstore.as_retriever()


# ==============================
# 5️⃣ Setup Gemini LLM
# ==============================

print("4. Setting up the AI Brain...")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3
)


# ==============================
# 6️⃣ Prompt Template
# ==============================

system_prompt = (
    "You are a highly intelligent legal assistant specializing in Bangladesh Law. "
    "Use the provided context to answer the user's question. "
    "If you don't know the answer based on the context, say "
    "'I cannot find the answer in the provided legal documents.' "
    "Do not make up any laws.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])


# ==============================
# 7️⃣ Create RAG Pipeline
# ==============================

question_answer_chain = create_stuff_documents_chain(
    llm,
    prompt
)

rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)


# ==============================
# 8️⃣ Test the System
# ==============================

print("\n--- SYSTEM READY ---")

user_question = "What are the working hours for an adult worker?"

print(f"\nUser: {user_question}")

response = rag_chain.invoke({
    "input": user_question
})

print("\nAI Assistant:")
print(response["answer"])