import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate 
import os

from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain.tools.tavily_search import TavilySearchResults


os.environ["GOOGLE_API_KEY"] = "" #add your google ai api key
os.environ["TAVILY_API_KEY"] = "" #add your tavily api key

model= ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.2)

search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search)

agent_chain = initialize_agent(
    [tavily_tool],
    model,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):

    model_name = "BAAI/bge-base-en"
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

    embeddings = HuggingFaceBgeEmbeddings (
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs=encode_kwargs)



    vectorstore = Chroma.from_texts(text_chunks, embeddings, persist_directory="./chroma_db")
    return vectorstore

def get_conversation_chain(vectorstore):
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the question is not related to the provided context,
    just say, "I don't know"
    Context:\n{context}?\n
    Question: \n{question}\n

    Answer:
    """
    model= ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.2)
    prompt = PromptTemplate(template = prompt_template,input_variables =["context","question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, agent_chain):

    model_name = "BAAI/bge-base-en"
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

    embeddings = HuggingFaceBgeEmbeddings (
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs=encode_kwargs)

    new_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversation_chain(new_db)

    with st.chat_message("assistant"):
        stream = chain(
        {"input_documents":docs, "question" : user_question}, return_only_outputs=True)
        response = stream["output_text"]
        if response.strip() =="I don't know":
            response= agent_chain.run(user_question,)
    return response

def main(agent_chain):
    load_dotenv()
    st.set_page_config(page_title="Perplexity-Mini", page_icon=":books:")
    
    with st.sidebar:
        st.header("Chat with your PDFs :books:")

        with st.sidebar:
            st.title("Menu: ")
            # Upload documents
            pdf_docs = st.file_uploader("Upload your PDF docs here", accept_multiple_files=True, type=['pdf'])
            if st.button("Process"):
                with st.spinner("Adding Knowledge base"):
                    raw_text =get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.success("Done")
    
    user_question = st.text_input("Ask a Question from the PDF Files")

    
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Add new question to history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


    # Display user message in chat message container

    if user_question:
        with st.chat_message("user"):
            st.write(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})


        response= user_input(user_question, agent_chain)
        #response
        with st.chat_message("assistant"):
            st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})



if __name__=="__main__":
    main(agent_chain)