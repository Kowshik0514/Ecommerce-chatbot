import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from knowledge_loader import get_combined_knowledge
from langchain.schema import Document

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = google_api_key

@st.cache_resource
def initialize_chain():
    product_file, faq_documents = get_combined_knowledge()
    loader = TextLoader(product_file)
    product_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    product_chunks = text_splitter.split_documents(product_documents)
    faq_docs = [Document(page_content=faq) for faq in faq_documents]

    all_documents = product_chunks + faq_docs
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma.from_documents(all_documents, embeddings, persist_directory="./chroma_db")
    template = """You are a helpful customer support assistant for an electronics store called TechX.
Use the following context to answer the customer's question.
If you don't know the answer, say you don't know and offer to connect them with a human agent.
Be friendly, concise, and professional.

Context: {context}

Chat History: {chat_history}

Customer Question: {question}

Answer:"""
    PROMPT = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=template,
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )
    return qa_chain

st.set_page_config(page_title="TechX Customer Support", page_icon=":robot_face:")
st.title("TechX Customer Support")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to TechX Customer Support! How can I help you today?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Ask about our products, orders, or policies...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
    qa_chain = initialize_chain()
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = qa_chain({"question": user_query})
            st.markdown(response["answer"])
    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

if __name__ == "__main__":
    st.sidebar.title("About")
    st.sidebar.info(
        "This is a RAG-based customer support chatbot "
        "that uses LangChain and ChromaDB to provide answers "
        "about products, policies, and frequently asked questions."
    )
