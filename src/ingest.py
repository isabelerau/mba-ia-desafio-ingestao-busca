import os
import time
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from search import get_store

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
for k in ("GOOGLE_API_KEY", "GOOGLE_EMBEDDING_MODEL", "DATABASE_URL", "PG_VECTOR_COLLECTION_NAME", "PDF_PATH"):
    if not os.getenv(k):
        raise ValueError(f"{k} is not set in the environment variables.")

PDF_PATH = ROOT / os.getenv("PDF_PATH")

BATCH_SIZE = 50
SLEEP_BETWEEN_BATCHES = 0  # aumentar se o free tier do Gemini der 429
MAX_RETRIES = 5


def ingest_pdf():
    docs = PyPDFLoader(str(PDF_PATH)).load()

    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    ).split_documents(docs)
    if not splits:
        print(f"No chunks extracted from {PDF_PATH}. Nothing to ingest.")
        return

    cleaned_docs = [
        Document(
            page_content=doc.page_content,
            metadata={k: v for k, v in doc.metadata.items() if v not in ("", None)}
        )
        for doc in splits
    ]

    # IDs determinísticos: reexecutar a ingestão sobrescreve em vez de duplicar
    ids = [f"doc:{i}" for i in range(len(cleaned_docs))]

    store = get_store()

    total = len(cleaned_docs)
    print(f"Total chunks to ingest: {total}")

    for batch_num, i in enumerate(range(0, total, BATCH_SIZE), start=1):
        batch_docs = cleaned_docs[i:i + BATCH_SIZE]
        batch_ids = ids[i:i + BATCH_SIZE]
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"Batch {batch_num} ({i + 1}-{i + len(batch_docs)} of {total})...")
                store.add_documents(batch_docs, ids=batch_ids)
                break
            except Exception as e:
                if attempt == MAX_RETRIES:
                    raise
                wait = 60 * attempt
                print(f"  Failed (attempt {attempt}): {type(e).__name__}. Sleeping {wait}s and retrying...")
                time.sleep(wait)
        if i + BATCH_SIZE < total:
            time.sleep(SLEEP_BETWEEN_BATCHES)

    print(f"Done. {total} chunks ingested into '{os.getenv('PG_VECTOR_COLLECTION_NAME')}'.")


if __name__ == "__main__":
    ingest_pdf()
