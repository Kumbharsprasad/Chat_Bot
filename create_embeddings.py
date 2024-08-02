import pickle
import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_embeddings():
    # Initialize embeddings model
    print("Initializing embeddings model...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load documents
    print("Loading documents from directory...")
    loader = PyPDFDirectoryLoader("./text_shoot")  # Data Ingestion
    docs = loader.load()  # Document Loading
    
    # Check if documents are loaded
    if not docs:
        print("No documents found. Please check the directory.")
        return
    
    print(f"Loaded {len(docs)} documents.")
    
    # Split documents into chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
    final_documents = text_splitter.split_documents(docs[:20])  # Splitting
    
    # Check if documents are split properly
    if not final_documents:
        print("Document splitting failed. Please check the text_splitter configuration.")
        return
    
    print(f"Created {len(final_documents)} document chunks.")
    
    # Create vector embeddings
    print("Creating vector embeddings...")
    vectors = FAISS.from_documents(final_documents, embeddings)  # Vector embeddings
    
    # Save embeddings to a file
    print("Saving embeddings to vectors.pkl...")
    with open("vectors.pkl", "wb") as f:
        pickle.dump(vectors, f)
    
    # Verify the file size
    file_size = os.path.getsize("vectors.pkl")
    if file_size > 0:
        print(f"Embeddings successfully saved to vectors.pkl (size: {file_size} bytes).")
    else:
        print("Failed to save embeddings: vectors.pkl is empty.")

if __name__ == "__main__":
    create_embeddings()
