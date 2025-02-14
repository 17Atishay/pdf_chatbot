from pypdf import PdfReader
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from dotenv import load_dotenv
import pickle
import os


# sidebar contents
with st.sidebar:
    st.title("LLM Chat App")
    st.markdown(''' 
    ## About
    This App is an LLM-powered Chatbot built using:
    
    ''')
    st.write('\n' * 5)
    st.write("Made with Love by Atishay Jain")

def main():
    st.header("Chat with pdf-bot")

    load_dotenv()

    # upload a pdf file
    pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
    # st.write(pdf.name)

    # st.write(pdf)
    if pdf:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        formatted_chunks = [Document(page_content=text) for text in chunks]
        st.write(formatted_chunks)
        
        # Step 3: Generate embeddings
        # embeddings = HuggingFaceEmbeddings()

        # Step 4: Create a vector database
        # vector_db = FAISS.from_documents(formatted_chunks, embeddings)
        store_name = pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"):  # Here we check if the embedded file exists
            with open(f"{store_name}.pkl", "rb") as f:
                vector_db = pickle.load(f)
            st.write("Embeddings loaded from file")
        else:
            embeddings = HuggingFaceEmbeddings()
            vector_db = FAISS.from_documents(formatted_chunks, embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vector_db, f)
            st.write("Embeddings computation completed.")

        # Step 5: Implement open-source LLM model
        # Accept user input questions/query
        query = st.text_input("Ask questions about your pdf data: ")
        # st.write(query)
        
        if(query):
            docs = vector_db.similarity_search(query,k=2)
            # st.write(docs)

            # Retrieve the HuggingFace Hub API token
            HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
            if not HUGGINGFACE_API_TOKEN:
                raise ValueError("HuggingFace Hub API token is missing. Check your .env file.")
            llm = HuggingFaceHub(
                repo_id="google/flan-t5-large",  # Replace with the desired model
                model_kwargs={"temperature": 0.2, "max_length": 512},
                huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
            )

            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            st.write(response)

if __name__ == '__main__':
    main()
