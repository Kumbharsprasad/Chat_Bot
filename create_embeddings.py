import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_embeddings():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    loader = PyPDFDirectoryLoader("./text_shoot")  # Data Ingestion
    docs = loader.load()  # Document Loading
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
    final_documents = text_splitter.split_documents(docs[:20])  # Splitting
    vectors = FAISS.from_documents(final_documents, embeddings)  # Vector embeddings
    
    with open("vectors.pkl", "wb") as f:
        pickle.dump(vectors, f)
    print("Embeddings created and saved to vectors.pkl")

if __name__ == "__main__":
    create_embeddings()
