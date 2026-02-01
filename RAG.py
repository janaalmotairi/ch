import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

def setup_vector_db(path):

    #Reads the CSV file, creates embeddings, and saves the FAISS index locally.

    try:
        # 1. Read the CSV file
        df = pd.read_csv(path, encoding='utf-8')
    except Exception as e:
        return None, e
    
    documents = []

    # 2. Convert each row in the DataFrame into a text document
    for index, row in df.iterrows():
        # Join column names and values into a single string per row
        content = "|".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
        documents.append(Document(page_content=content))

    # 3. Initialize the Embedding Model (converts text to vectors)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Create the FAISS Vector Database from documents
    vector_db = FAISS.from_documents(documents, embeddings)

    # 5. Save the vector database locally to be used by the chatbot
    vector_db.save_local("faiss_index_hr")
    
    return vector_db, "done"


def get_context(query):
    
    #Loads the local vector DB and retrieves relevant context based on the user's query.
    
    # Initialize the same embedding model used for creation
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    try:
        # 1. Load the locally saved FAISS index
        # allow_dangerous_deserialization=True is required when loading local pickle files
        vector_db = FAISS.load_local("faiss_index_hr", embeddings, allow_dangerous_deserialization=True)
        
        # 2. Perform similarity search (retrieve top 20 relevant results)
        results = vector_db.similarity_search(query, k=20)
        
        # 3. Combine the results into a single string
        context_text = "\n".join([res.page_content for res in results])
        
        return context_text
    except:
        # Return empty string if no index is found or an error occurs
        return ""

# --- Main execution block ---
if __name__ == "__main__":
    # Define the dataset file name
    file_name = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    
    # Run the setup function to create the database
    setup_vector_db(file_name)