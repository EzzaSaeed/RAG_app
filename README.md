 RAG PDF Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that enables users to upload PDF documents and ask questions based on their content. The application retrieves relevant information from the uploaded document using embeddings and vector search before generating accurate, context-aware responses with a Large Language Model (LLM).

 Features

- Upload PDF documents
- Extract and process text automatically
- Split documents into semantic chunks
- Generate embeddings using Sentence Transformers
- Store embeddings in a FAISS vector database
- Retrieve the most relevant document chunks
- Answer questions using OpenRouter LLM
- Simple and interactive Gradio interface
- Deployable on Hugging Face Spaces

 Tech Stack

- Python
- Gradio
- LangChain
- FAISS
- Sentence Transformers
- OpenRouter API
- Hugging Face Spaces
- PyPDF

 Project Workflow

1. Upload a PDF document.
2. Extract text from the PDF.
3. Split the text into chunks.
4. Generate embeddings.
5. Store embeddings in a FAISS vector database.
6. Retrieve relevant document chunks for each query.
7. Generate grounded responses using an LLM via OpenRouter.

 Project Highlights

- Implements Retrieval-Augmented Generation (RAG)
- Uses vector similarity search for document retrieval
- Reduces hallucinations by grounding responses in uploaded documents
- Compares document-grounded answers with plain LLM responses
- Demonstrates practical use of embeddings and vector databases

```

 Demo

Upload a PDF and ask questions such as:

- Summarize this document.
- What are the main objectives?
- Who is the intended audience?
- List the important findings.
- What conclusions are presented?

Repository Structure

```
.
├── app.py
├── requirements.txt
├── README.md
└── assets/
```

Learning Outcomes

This project helped me gain practical experience with:

- Retrieval-Augmented Generation (RAG)
- Large Language Models (LLMs)
- Embeddings
- Vector Databases (FAISS)
- Prompt Engineering
- LangChain
- Hugging Face Spaces
- OpenRouter API Integration



 Live demo:
https://ezzasaeed-rag-app.hf.space/
