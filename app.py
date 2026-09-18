"""
Assignment 39 deployment target:
Assignment 30 - ChatGroq Document RAG.

Deploy this file on:
- Streamlit Community Cloud
- Hugging Face Spaces (Docker)
"""

import os

import streamlit as st


def configure_streamlit_secrets():
    """
    Streamlit Community Cloud stores secrets in st.secrets.
    Hugging Face Spaces normally exposes Space secrets as environment vars.

    This copies supported Streamlit secrets into os.environ before importing
    the RAG backend.
    """
    supported = [
        "GROQ_API_KEY",
        "GROQ_MODEL",
        "HF_EMBEDDING_MODEL",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "TOP_K",
        "MAX_HISTORY_MESSAGES",
    ]

    for key in supported:
        if os.getenv(key):
            continue

        try:
            value = st.secrets.get(
                key
            )
        except Exception:
            value = None

        if value is not None:
            os.environ[key] = str(
                value
            )


configure_streamlit_secrets()

from rag_backend import (  # noqa: E402
    EMBEDDING_MODEL,
    GROQ_MODEL,
    build_uploaded_assistant,
    upload_signature,
)


st.set_page_config(
    page_title="ChatGroq Document RAG",
    page_icon="⚡",
    layout="centered",
)

st.title(
    "⚡ ChatGroq Document RAG"
)

st.caption(
    f"Groq: {GROQ_MODEL} • "
    f"Embeddings: {EMBEDDING_MODEL} • "
    "FAISS"
)

st.write(
    "Upload PDF or TXT documents and ask grounded questions. "
    "The app is the deployment version of Assignment 30."
)


with st.sidebar:
    st.header(
        "Documents"
    )

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT",
        type=[
            "pdf",
            "txt",
        ],
        accept_multiple_files=True,
    )

    if st.button(
        "Clear conversation"
    ):
        if (
            "rag_assistant"
            in st.session_state
            and st.session_state.rag_assistant
            is not None
        ):
            st.session_state.rag_assistant.clear_history()

        st.session_state.messages = []
        st.rerun()

    st.info(
        "Deployment version: Groq generates answers, "
        "HuggingFace creates embeddings, and FAISS performs retrieval."
    )


if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_signature" not in st.session_state:
    st.session_state.document_signature = None

if "rag_assistant" not in st.session_state:
    st.session_state.rag_assistant = None

if "document_stats" not in st.session_state:
    st.session_state.document_stats = None


if uploaded_files:
    signature = upload_signature(
        uploaded_files
    )

    if (
        signature
        != st.session_state.document_signature
    ):
        try:
            with st.spinner(
                "Loading, chunking and indexing documents..."
            ):
                (
                    documents,
                    chunks,
                    _vectorstore,
                    assistant,
                ) = build_uploaded_assistant(
                    uploaded_files
                )

            st.session_state.document_signature = (
                signature
            )
            st.session_state.rag_assistant = (
                assistant
            )
            st.session_state.document_stats = {
                "documents":
                    len(documents),
                "chunks":
                    len(chunks),
                "files":
                    len(uploaded_files),
            }

            st.session_state.messages = []

            st.success(
                "Documents indexed successfully."
            )

        except Exception as error:
            st.session_state.rag_assistant = None

            st.error(
                "Could not prepare the RAG knowledge base."
            )

            st.exception(
                error
            )


if st.session_state.document_stats:
    stats = (
        st.session_state.document_stats
    )

    st.caption(
        f"Files: {stats['files']} • "
        f"Loaded docs/pages: {stats['documents']} • "
        f"Chunks: {stats['chunks']}"
    )


for message in st.session_state.messages:
    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )

        if message.get(
            "latency_ms"
        ) is not None:
            st.caption(
                "Observed RAG latency: "
                f"{message['latency_ms']:.0f} ms"
            )


question = st.chat_input(
    "Ask about the uploaded documents...",
    disabled=(
        st.session_state.rag_assistant
        is None
    ),
)


if question:
    st.session_state.messages.append(
        {
            "role":
                "user",
            "content":
                question,
        }
    )

    with st.chat_message(
        "user"
    ):
        st.markdown(
            question
        )

    with st.chat_message(
        "assistant"
    ):
        try:
            with st.spinner(
                "Retrieving context and asking Groq..."
            ):
                result = (
                    st.session_state
                    .rag_assistant
                    .ask(
                        question
                    )
                )

            answer = result[
                "answer"
            ]

            st.markdown(
                answer
            )

            st.caption(
                "Observed RAG latency: "
                f"{result['latency_ms']:.0f} ms"
            )

            with st.expander(
                "Retrieved evidence"
            ):
                st.write(
                    "Standalone search query:",
                    result[
                        "search_query"
                    ],
                )

                for index, doc in enumerate(
                    result[
                        "documents"
                    ],
                    start=1,
                ):
                    meta = (
                        doc.metadata
                        or {}
                    )

                    source = (
                        meta.get(
                            "uploaded_file"
                        )
                        or meta.get(
                            "source"
                        )
                        or "document"
                    )

                    page = meta.get(
                        "page"
                    )

                    heading = (
                        f"Source {index}: "
                        f"{source}"
                    )

                    if page is not None:
                        heading += (
                            f" | page index {page}"
                        )

                    st.markdown(
                        f"**{heading}**"
                    )

                    st.write(
                        doc.page_content[
                            :1200
                        ]
                    )

            st.session_state.messages.append(
                {
                    "role":
                        "assistant",
                    "content":
                        answer,
                    "latency_ms":
                        result[
                            "latency_ms"
                        ],
                }
            )

        except Exception as error:
            st.error(
                "The RAG request failed."
            )
            st.exception(
                error
            )


if not uploaded_files:
    st.info(
        "Upload at least one PDF or TXT file to start the chatbot."
    )
