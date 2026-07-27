import spaces
import os
import gradio as gr
from pypdf import PdfReader

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from openai import OpenAI

# ==========================
# OpenRouter
# ==========================

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

MODEL_NAME = "meta-llama/llama-3.3-70b-instruct"

# Embedding model
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Global variables
index = None
chunks = []
# ==========================
# Read PDF
# ==========================

def load_pdf(pdf_path):

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ==========================
# Split Text
# ==========================

def split_text(text, chunk_size=800, overlap=100):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks


# ==========================
# Build FAISS Index
# ==========================
import spaces

@spaces.GPU(duration=60)  
def build_index(text):

    chunks = split_text(text)

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, chunks


# ==========================
# Process PDF
# ==========================

def process_pdf(pdf):

    global index
    global chunks

    text = load_pdf(pdf)

    index, chunks = build_index(text)

    return f"✅ PDF processed successfully! ({len(chunks)} chunks created)"
        # ==========================
# Ask Question
# ==========================

def ask_question(question):

    global index
    global chunks

    if index is None:
        return "❌ Please upload and process a PDF first."

    if question.strip() == "":
        return "❌ Please enter a question."

    try:

        # Create embedding for the question
        query_embedding = embedding_model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        # Search top 3 similar chunks
        k = 3
        scores, indices = index.search(query_embedding, k)

        context = ""

        for i in indices[0]:
            if i < len(chunks):
                context += chunks[i] + "\n\n"

        prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the context below.

If the answer is not present in the context, reply exactly:

"I couldn't find that information in the uploaded PDF."

Context:
{context}

Question:
{question}
"""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        return response.choices[0].message.content

    except Exception as e:

        return f"❌ Error:\n{str(e)}"
        # ==========================
# Gradio UI
# ==========================

with gr.Blocks(title="RAG PDF Chatbot") as demo:

    gr.Markdown("# 📄 RAG PDF Chatbot")
    gr.Markdown(
        "Upload a PDF and ask questions using Retrieval-Augmented Generation (RAG)."
    )

    pdf = gr.File(
        label="Upload PDF",
        file_types=[".pdf"],
        type="filepath"
    )

    process_btn = gr.Button("📄 Process PDF", variant="primary")

    status = gr.Textbox(
        label="Status",
        interactive=False
    )

    process_btn.click(
        fn=process_pdf,
        inputs=pdf,
        outputs=status
    )

    gr.Markdown("## Ask Questions")

    question = gr.Textbox(
        label="Question",
        placeholder="Ask anything about the uploaded PDF..."
    )

    ask_btn = gr.Button("Ask", variant="primary")

    answer = gr.Textbox(
        label="Answer",
        lines=10
    )

    ask_btn.click(
        fn=ask_question,
        inputs=question,
        outputs=answer
    )

    question.submit(
        fn=ask_question,
        inputs=question,
        outputs=answer
    )

demo.queue()
demo.launch()
