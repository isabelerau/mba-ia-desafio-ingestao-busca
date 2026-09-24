import os
import logging
import warnings
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Evita que avisos das bibliotecas do Gemini poluam a saída do chat
warnings.filterwarnings("ignore", message=".*sampling parameter.*will be ignored")
logging.getLogger("google_genai").setLevel(logging.ERROR)

REQUIRED_VARS = ("GOOGLE_API_KEY", "GOOGLE_EMBEDDING_MODEL", "GOOGLE_LLM_MODEL", "DATABASE_URL", "PG_VECTOR_COLLECTION_NAME")

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model=os.getenv("GOOGLE_EMBEDDING_MODEL"))

def get_store():
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )

def search_prompt(question=None):
    missing = [k for k in REQUIRED_VARS if not os.getenv(k)]
    if missing:
        print(f"Variáveis de ambiente não definidas: {', '.join(missing)}")
        return None

    # Criado uma única vez e reaproveitado em todas as perguntas
    store = get_store()

    def buscar_contexto(pergunta):
        # Vetoriza a pergunta e busca os 10 chunks mais relevantes
        resultados = store.similarity_search_with_score(pergunta, k=10)
        return "\n\n".join(doc.page_content.strip() for doc, _score in resultados)

    llm = ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_LLM_MODEL"), temperature=0)
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = (
        {"contexto": RunnableLambda(buscar_contexto), "pergunta": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    if question is not None:
        return chain.invoke(question)
    return chain