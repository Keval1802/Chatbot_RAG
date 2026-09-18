"""
Deployment version of Assignment 30: ChatGroq Document RAG.

Pipeline:
PDF/TXT upload
    -> LangChain loaders
    -> RecursiveCharacterTextSplitter
    -> HuggingFace embeddings
    -> FAISS
    -> history-aware retriever query
    -> ChatGroq
    -> grounded answer

Why HuggingFace embeddings here instead of local Ollama embeddings?
Streamlit Community Cloud and Hugging Face Spaces cannot directly use the
Ollama server running on the student's own laptop. HuggingFace embeddings
run inside the hosted Python process and keep the app deployable.
"""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Iterable, Sequence
import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
)

EMBEDDING_MODEL = os.getenv(
    "HF_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", "1000")
)

CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", "150")
)

TOP_K = int(
    os.getenv("TOP_K", "4")
)

MAX_HISTORY_MESSAGES = int(
    os.getenv("MAX_HISTORY_MESSAGES", "8")
)


def require_groq_key() -> None:
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not configured. "
            "Add it to Streamlit Cloud Secrets or Hugging Face Space Secrets."
        )


def get_llm() -> ChatGroq:
    require_groq_key()

    return ChatGroq(
        model=os.getenv(
            "GROQ_MODEL",
            GROQ_MODEL,
        ),
        temperature=0.1,
        max_tokens=800,
    )


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Cache one embedding model per hosted Python process.
    """
    return HuggingFaceEmbeddings(
        model_name=os.getenv(
            "HF_EMBEDDING_MODEL",
            EMBEDDING_MODEL,
        ),
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )


def _uploaded_name_and_bytes(uploaded, index: int) -> tuple[str, bytes]:
    """
    Support:
    - Streamlit UploadedFile objects
    - Gradio file path strings
    - pathlib.Path objects
    """
    if hasattr(uploaded, "getvalue"):
        file_name = Path(
            getattr(
                uploaded,
                "name",
                f"upload_{index}",
            )
        ).name
        return file_name, uploaded.getvalue()

    file_path = Path(str(uploaded))
    if not file_path.exists():
        raise FileNotFoundError(
            f"Uploaded file was not found: {file_path}"
        )

    return file_path.name, file_path.read_bytes()


def load_uploaded_files(
    uploaded_files: Sequence,
) -> list[Document]:
    """
    Load PDF/TXT inputs from Streamlit or Gradio.
    """
    documents: list[Document] = []

    with TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)

        for index, uploaded in enumerate(
            uploaded_files,
            start=1,
        ):
            file_name, file_bytes = _uploaded_name_and_bytes(
                uploaded,
                index,
            )

            suffix = Path(file_name).suffix.lower()

            if suffix not in {".pdf", ".txt"}:
                raise ValueError(
                    f"Unsupported file type: {file_name}. "
                    "Only PDF and TXT are supported."
                )

            temp_path = temp_root / file_name
            temp_path.write_bytes(file_bytes)

            if suffix == ".pdf":
                loaded = PyPDFLoader(
                    str(temp_path)
                ).load()
            else:
                loaded = TextLoader(
                    str(temp_path),
                    encoding="utf-8",
                ).load()

            for doc in loaded:
                doc.metadata = dict(
                    doc.metadata or {}
                )
                doc.metadata[
                    "uploaded_file"
                ] = file_name

            documents.extend(loaded)

    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(
        documents
    )

    cleaned: list[Document] = []

    for index, chunk in enumerate(
        chunks
    ):
        text = (
            chunk.page_content
            or ""
        ).strip()

        if not text:
            continue

        chunk.page_content = text
        chunk.metadata = dict(
            chunk.metadata or {}
        )
        chunk.metadata[
            "chunk_id"
        ] = index

        cleaned.append(
            chunk
        )

    return cleaned


def create_vectorstore(
    chunks: list[Document],
) -> FAISS:
    if not chunks:
        raise ValueError(
            "No text chunks were created from the uploaded documents."
        )

    return FAISS.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
    )


def create_retriever(
    vectorstore: FAISS,
    top_k: int = TOP_K,
):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": top_k,
        },
    )


RAG_SYSTEM_INSTRUCTION = """
You are a document-grounded Q&A assistant.

Rules:
1. Answer factual questions ONLY from the retrieved document context below.
2. Do not invent facts or use outside knowledge to fill gaps.
3. If the retrieved context does not contain enough information, reply exactly:
   "I don't know based on the uploaded documents."
4. Use conversation history only to understand references and follow-up
   questions. Factual claims must still be supported by retrieved context.
5. Keep the answer clear and concise.

Retrieved context:
{context}
""".strip()


RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            RAG_SYSTEM_INSTRUCTION,
        ),
        MessagesPlaceholder(
            variable_name="history",
        ),
        (
            "human",
            "{question}",
        ),
    ]
)


CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Rewrite the current user message as a standalone document-search query.

Use recent conversation history only to resolve references such as:
- "Explain more"
- "What about that?"
- "What was the previous point?"

Rules:
- Do not answer the question.
- Do not add facts.
- If already standalone, return it unchanged.
- Return only the standalone search query.
""".strip(),
        ),
        MessagesPlaceholder(
            variable_name="history",
        ),
        (
            "human",
            "{question}",
        ),
    ]
)


def trim_history(
    history: list[BaseMessage],
    max_messages: int = MAX_HISTORY_MESSAGES,
) -> list[BaseMessage]:
    if max_messages <= 0:
        return []

    return list(
        history[-max_messages:]
    )


def format_documents(
    documents: Iterable[Document],
) -> str:
    parts: list[str] = []

    for index, doc in enumerate(
        documents,
        start=1,
    ):
        meta = doc.metadata or {}

        source = (
            meta.get("uploaded_file")
            or meta.get("source")
            or "document"
        )

        page = meta.get("page")

        heading = (
            f"[Source {index}] {source}"
        )

        if page is not None:
            heading += (
                f" | page_index={page}"
            )

        parts.append(
            f"{heading}\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(
        parts
    )


def upload_signature(
    uploaded_files: Sequence,
) -> str:
    digest = sha256()

    for index, uploaded in enumerate(
        uploaded_files,
        start=1,
    ):
        file_name, file_bytes = _uploaded_name_and_bytes(
            uploaded,
            index,
        )

        digest.update(
            file_name.encode("utf-8")
        )
        digest.update(
            file_bytes
        )

    return digest.hexdigest()


class ConversationalGroqRAG:
    def __init__(
        self,
        vectorstore: FAISS,
        max_history_messages: int = MAX_HISTORY_MESSAGES,
        top_k: int = TOP_K,
    ):
        self.vectorstore = vectorstore

        self.retriever = create_retriever(
            vectorstore,
            top_k=top_k,
        )

        self.llm = get_llm()

        self.answer_chain = (
            RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )

        self.contextualize_chain = (
            CONTEXTUALIZE_PROMPT
            | self.llm
            | StrOutputParser()
        )

        self.history: list[
            BaseMessage
        ] = []

        self.max_history_messages = (
            max_history_messages
        )

    def _recent_history(
        self,
    ) -> list[BaseMessage]:
        return trim_history(
            self.history,
            self.max_history_messages,
        )

    def clear_history(
        self,
    ) -> None:
        self.history = []

    def standalone_query(
        self,
        question: str,
    ) -> str:
        recent = self._recent_history()

        if not recent:
            return question.strip()

        rewritten = (
            self.contextualize_chain.invoke(
                {
                    "history":
                        recent,
                    "question":
                        question,
                }
            )
        )

        return (
            rewritten.strip()
            or question.strip()
        )

    def ask(
        self,
        question: str,
    ) -> dict:
        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        start = perf_counter()

        search_query = (
            self.standalone_query(
                question
            )
        )

        documents = self.retriever.invoke(
            search_query
        )

        context = format_documents(
            documents
        )

        answer = self.answer_chain.invoke(
            {
                "context":
                    context,
                "history":
                    self._recent_history(),
                "question":
                    question,
            }
        )

        self.history.extend(
            [
                HumanMessage(
                    content=question
                ),
                AIMessage(
                    content=answer
                ),
            ]
        )

        self.history = trim_history(
            self.history,
            self.max_history_messages,
        )

        latency_ms = (
            perf_counter()
            - start
        ) * 1000

        return {
            "answer":
                answer,
            "search_query":
                search_query,
            "documents":
                documents,
            "latency_ms":
                latency_ms,
        }


def build_uploaded_assistant(
    uploaded_files: Sequence,
):
    documents = load_uploaded_files(
        uploaded_files
    )

    chunks = split_documents(
        documents
    )

    vectorstore = create_vectorstore(
        chunks
    )

    assistant = (
        ConversationalGroqRAG(
            vectorstore
        )
    )

    return (
        documents,
        chunks,
        vectorstore,
        assistant,
    )
