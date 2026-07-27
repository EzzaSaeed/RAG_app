import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="RAG PDF Chatbot")

st.title("📄 RAG PDF Chatbot")
st.write("Upload a PDF and ask questions about its content.")

api_key = st.secrets["OPENROUTER_API_KEY"]

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k":3})

    llm = ChatOpenAI(
        model="openai/gpt-4.1-mini",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's question using only the provided context. "
                "If the answer is not in the document, say 'I don't know.'\n\n"
                "Context:\n{context}",
            ),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    rag_chain = create_retrieval_chain(
        retriever,
        question_answer_chain
    )

    question = st.text_input("Ask a question")

    if question:

        response = rag_chain.invoke(
            {"input": question}
        )

        st.subheader("Answer")
        st.write(response["answer"])

        st.subheader("Retrieved Chunks")

        for doc in response["context"]:
            st.info(doc.page_content[:500])
