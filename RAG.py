import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

def setup_vector_db(path):

    try:
        df=pd.read_csv(path,encoding='utf-8')
    except Exception as e:
        return None, e
    

    documents=[]

    for index, row in df.iterrows():
        content="|".join([f"{col}: {val}" for col, val in row.items()
                        if pd.notna(val)])
        documents.append(Document(page_content=content))


    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


    vector_db=FAISS.from_documents(documents,embeddings)

    vector_db.save_local("faiss_index_hr")
    return vector_db, "done"



def get_context(query):
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    try:
        vector_db=FAISS.load_local("faiss_index_hr",embeddings,allow_dangerous_deserialization=True)
        results=vector_db.similarity_search(query,k=3)
        context_text="\n".join([res.page_content for res in results])
        return context_text
    except:
        return ""




if __name__=="__main__":
    file_name="WA_Fn-UseC_-HR-Employee-Attrition.csv"
    setup_vector_db(file_name)