"""
Hugging Face Spaces entry point for Assignment 39.

Free-deployment path:
- Gradio SDK
- ZeroGPU hardware (for eligible free personal accounts)
- ChatGroq API for generation
- HuggingFace embeddings + FAISS for retrieval

This app does not require Docker.
"""

import os

import gradio as gr

from rag_backend import (
    EMBEDDING_MODEL,
    GROQ_MODEL,
    build_uploaded_assistant,
)


def prepare_documents(files):
    if not files:
        return None, "Upload at least one PDF or TXT file first."

    try:
        (
            documents,
            chunks,
            _vectorstore,
            assistant,
        ) = build_uploaded_assistant(files)

        status = (
            f"Ready. Files: {len(files)} | "
            f"Loaded docs/pages: {len(documents)} | "
            f"Chunks: {len(chunks)}"
        )

        return assistant, status

    except Exception as error:
        return None, f"Indexing failed: {error}"


def answer_message(
    message,
    history,
    assistant,
):
    history = history or []

    if assistant is None:
        history.append(
            {
                "role": "assistant",
                "content": (
                    "Please upload PDF/TXT documents and click "
                    "'Prepare documents' first."
                ),
            }
        )
        return "", history, assistant

    if not message or not message.strip():
        return "", history, assistant

    history.append(
        {
            "role": "user",
            "content": message,
        }
    )

    try:
        result = assistant.ask(message)

        answer = result["answer"]

        evidence_lines = []
        for index, doc in enumerate(
            result["documents"],
            start=1,
        ):
            meta = doc.metadata or {}
            source = (
                meta.get("uploaded_file")
                or meta.get("source")
                or "document"
            )
            page = meta.get("page")

            label = f"Source {index}: {source}"
            if page is not None:
                label += f" | page index {page}"

            evidence_lines.append(
                f"{label}\n{doc.page_content[:500]}"
            )

        if evidence_lines:
            answer += (
                "\n\n---\n\nRetrieved evidence:\n\n"
                + "\n\n".join(evidence_lines)
            )

        history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

    except Exception as error:
        history.append(
            {
                "role": "assistant",
                "content": f"RAG request failed: {error}",
            }
        )

    return "", history, assistant


def clear_all():
    return None, "No documents prepared.", []


with gr.Blocks(
    title="ChatGroq Document RAG"
) as demo:
    gr.Markdown(
        f"""
# ⚡ ChatGroq Document RAG

**Groq model:** `{GROQ_MODEL}`  
**Embeddings:** `{EMBEDDING_MODEL}`  
**Retriever:** FAISS

Upload PDF/TXT documents, prepare the knowledge base, and ask grounded
questions.
"""
    )

    assistant_state = gr.State(
        value=None
    )

    files = gr.File(
        label="Upload PDF or TXT documents",
        file_count="multiple",
        file_types=[
            ".pdf",
            ".txt",
        ],
        type="filepath",
    )

    with gr.Row():
        prepare_button = gr.Button(
            "Prepare documents",
            variant="primary",
        )
        clear_button = gr.Button(
            "Clear"
        )

    status = gr.Textbox(
        value="No documents prepared.",
        label="Status",
        interactive=False,
    )

    chatbot = gr.Chatbot(
        label="Chat",
        height=500,
    )

    message = gr.Textbox(
        label="Question",
        placeholder="Ask about the uploaded documents...",
    )

    send_button = gr.Button(
        "Send",
        variant="primary",
    )

    prepare_button.click(
        fn=prepare_documents,
        inputs=[
            files,
        ],
        outputs=[
            assistant_state,
            status,
        ],
    )

    send_button.click(
        fn=answer_message,
        inputs=[
            message,
            chatbot,
            assistant_state,
        ],
        outputs=[
            message,
            chatbot,
            assistant_state,
        ],
    )

    message.submit(
        fn=answer_message,
        inputs=[
            message,
            chatbot,
            assistant_state,
        ],
        outputs=[
            message,
            chatbot,
            assistant_state,
        ],
    )

    clear_button.click(
        fn=clear_all,
        outputs=[
            assistant_state,
            status,
            chatbot,
        ],
    )


if __name__ == "__main__":
    demo.launch()
