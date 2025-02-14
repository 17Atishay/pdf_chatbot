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
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer


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
    
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            # length_function=len   # Measures length by characters
            length_function=lambda x: len(tokenizer.encode(x))  # Measures length by tokens

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
            docs = vector_db.similarity_search(query,k=3)
            # st.write(docs)

            # Retrieve the HuggingFace Hub API token
            HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
            if not HUGGINGFACE_API_TOKEN:
                raise ValueError("HuggingFace Hub API token is missing. Check your .env file.")
            llm = HuggingFaceHub(
                repo_id="google/flan-t5-large",  # Replace with the desired model
                model_kwargs={"temperature": 0.5, "max_length": 512},
                huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
            )

            qa_prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="""
                Given the following document extract, answer the question clearly.

                Context:
                {context}

                Question: {question}

                Answer in a well-structured manner with relevant details.
                """
            )

            # "stuff" simply concatenates text → May limit response length.
            # "map_reduce" If your document is large, it performs better.
            # "refine" for better long answers
            chain = load_qa_chain(llm, chain_type="stuff", prompt=qa_prompt)
            response = chain.run(input_documents=docs, question=query)
            st.write(response)

if __name__ == '__main__':
    main()
