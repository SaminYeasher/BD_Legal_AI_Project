from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_and_split_pdf(pdf_path):

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap = 50
    )

    return splitter.split_documents(documents)