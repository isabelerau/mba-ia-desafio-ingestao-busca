# Ingestão e Busca Semântica com LangChain e Postgres

Sistema de RAG (Retrieval-Augmented Generation) que lê um PDF, armazena seu conteúdo como vetores em um PostgreSQL com pgVector e permite fazer perguntas sobre ele por um chat no terminal. As respostas se baseiam **apenas** no conteúdo do PDF: se a informação não estiver no documento, o sistema responde que não tem informações para responder.

## Como funciona

**Ingestão** ([src/ingest.py](src/ingest.py))

1. Carrega o PDF com `PyPDFLoader`.
2. Divide o texto em chunks de 1000 caracteres com overlap de 150 (`RecursiveCharacterTextSplitter`).
3. Gera o embedding de cada chunk com o modelo de embeddings do Gemini.
4. Grava os vetores no PostgreSQL/pgVector (`PGVector`), em lotes de 50, com novas tentativas em caso de erro de limite da API.

Os chunks recebem IDs fixos (`doc:0`, `doc:1`, ...), então rodar a ingestão de novo sobrescreve os registros em vez de duplicá-los.

**Busca e chat** ([src/search.py](src/search.py) e [src/chat.py](src/chat.py))

1. O usuário digita uma pergunta no terminal.
2. A pergunta é vetorizada e os 10 chunks mais parecidos são buscados no banco (`similarity_search_with_score(pergunta, k=10)`).
3. Os chunks são concatenados como `CONTEXTO` no prompt, junto com regras que proíbem o uso de conhecimento externo.
4. A LLM do Gemini (com `temperature=0`) gera a resposta, que é exibida no terminal.

## Tecnologias

- Python 3 (desenvolvido com Python 3.14)
- LangChain (`langchain-community`, `langchain-text-splitters`, `langchain-postgres`, `langchain-google-genai`)
- PostgreSQL 17 + pgVector, executado via Docker Compose
- Google Gemini para embeddings e LLM

## Estrutura do projeto

```text
├── docker-compose.yml    # PostgreSQL + pgVector (cria a extensão "vector" automaticamente)
├── requirements.txt      # Dependências Python
├── .env.example          # Modelo das variáveis de ambiente
├── src/
│   ├── ingest.py         # Ingestão do PDF no banco vetorial
│   ├── search.py         # Busca semântica, prompt e chain com a LLM
│   └── chat.py           # Chat interativo no terminal
├── document.pdf          # PDF usado na ingestão
└── README.md
```

## Pré-requisitos

- Python 3.10 ou superior
- Docker e Docker Compose
- Uma API Key do Google Gemini, que pode ser criada no [Google AI Studio](https://aistudio.google.com/apikey)

## Como executar

### 1. Clonar o repositório

```bash
git clone https://github.com/isabelerau/mba-ia-desafio-ingestao-busca.git
cd mba-ia-desafio-ingestao-busca
```

### 2. Criar o ambiente virtual e instalar as dependências

```bash
python3 -m venv venv
source venv/bin/activate        # no Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configurar as variáveis de ambiente

Copie o arquivo de exemplo e preencha sua API Key:

```bash
cp .env.example .env
```

| Variável                    | Descrição                                      | Valor padrão                                                |
| --------------------------- | ---------------------------------------------- | ----------------------------------------------------------- |
| `GOOGLE_API_KEY`            | API Key do Google Gemini                       | _(obrigatório preencher)_                                   |
| `GOOGLE_EMBEDDING_MODEL`    | Modelo de embeddings                           | `gemini-embedding-001`                                      |
| `GOOGLE_LLM_MODEL`          | Modelo usado para gerar as respostas           | `gemini-3.5-flash-lite`                                     |
| `DATABASE_URL`              | Conexão com o Postgres do `docker-compose.yml` | `postgresql+psycopg://postgres:postgres@localhost:5432/rag` |
| `PG_VECTOR_COLLECTION_NAME` | Nome da collection de vetores                  | `documentos`                                                |
| `PDF_PATH`                  | Caminho do PDF, relativo à raiz do projeto     | `document.pdf`                                              |

### 4. Subir o banco de dados

```bash
docker compose up -d
```

Isso sobe o PostgreSQL com pgVector na porta `5432` e cria a extensão `vector` no banco `rag`. Para conferir se o container está saudável:

```bash
docker compose ps
```

### 5. Executar a ingestão do PDF

```bash
python src/ingest.py
```

Saída esperada (a quantidade de chunks depende do PDF):

```text
Total chunks to ingest: 67
Batch 1 (1-50 of 67)...
Batch 2 (51-67 of 67)...
Done. 67 chunks ingested into 'documentos'.
```

### 6. Rodar o chat

```bash
python src/chat.py
```

Digite suas perguntas e pressione Enter. Para sair, digite `sair` (ou `exit`/`quit`) ou use `Ctrl+C`.

## Exemplo de uso

```text
Faça sua pergunta (digite 'sair' para encerrar):

PERGUNTA: Quando a empresa Alfa Tecnologia Holding foi fundada?
RESPOSTA: 1950

PERGUNTA: Qual o faturamento dela?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.

PERGUNTA: Qual o faturamento da empresa Alfa Tecnologia Holding?
RESPOSTA: R$ 66.776.155,12

PERGUNTA: sair
```

## Problemas comuns

- **`... is not set in the environment variables` / `Variáveis de ambiente não definidas`**: o arquivo `.env` não existe na raiz do projeto ou alguma variável está vazia. Revise o passo 3.
- **Erro de conexão com o banco**: verifique se o container está rodando com `docker compose ps` e se a porta `5432` não está sendo usada por outro Postgres local.
- **Erro 429 (limite de requisições) na ingestão**: o script já espera e tenta de novo automaticamente. Se persistir, aumente `SLEEP_BETWEEN_BATCHES` em [src/ingest.py](src/ingest.py).
- **Erro de dimensão do vetor após trocar o modelo de embeddings**: a tabela é criada com a dimensão do primeiro modelo usado. Ao trocar de modelo, apague o volume do banco e refaça a ingestão:

  ```bash
  docker compose down -v
  docker compose up -d
  python src/ingest.py
  ```

## Encerrando

```bash
docker compose down        # para o banco e mantém os dados
docker compose down -v     # para o banco e apaga os dados
deactivate                 # sai do ambiente virtual
```
