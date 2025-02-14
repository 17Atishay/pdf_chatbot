from pypdf import PdfReader
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import pickle
import os
from langchain.prompts import PromptTemplate
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Sidebar contents
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

    # Upload a PDF file
    pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

    if pdf:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            length_function=lambda x: len(tokenizer.encode(x))  # Measures length by tokens
        )
        chunks = text_splitter.split_text(text=text)
        formatted_chunks = [Document(page_content=text) for text in chunks]
        st.write(formatted_chunks)
        
        # Step 3: Generate embeddings
        store_name = pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"):  # Check if embeddings file exists
            with open(f"{store_name}.pkl", "rb") as f:
                vector_db = pickle.load(f)
            st.write("Embeddings loaded from file")
        else:
            embeddings = HuggingFaceEmbeddings()
            vector_db = FAISS.from_documents(formatted_chunks, embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vector_db, f)
            st.write("Embeddings computation completed.")

        # Step 5: Implement open-source LLM model using Transformers
        query = st.text_input("Ask questions about your PDF data: ")

        if query:
            docs = vector_db.similarity_search(query, k=3)

            # Load LLM Model using Transformers
            model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,  # Optimized for GPU
                device_map="auto"  # Uses GPU if available
            )

            # Format input for the model
            context = "\n".join([doc.page_content for doc in docs])
            prompt_text = f"""
            Given the following document extract, answer the question clearly.

            Context:
            {context}

            Question: {query}

            Answer:
            """

            inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

            with torch.no_grad():
                output = model.generate(**inputs, max_length=512, temperature=0.5)

            response = tokenizer.decode(output[0], skip_special_tokens=True)
            st.write(response)

if __name__ == '__main__':
    main()
