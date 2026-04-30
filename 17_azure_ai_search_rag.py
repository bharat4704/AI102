"""
============================================================
AI-102 | Program 17 — Azure AI Search (RAG Pipeline)
Service : Azure AI Search + Azure OpenAI
Skill   : Implement knowledge retrieval solutions
============================================================
Features:
  • Create search index with vector fields
  • Index documents with embeddings
  • Keyword search
  • Vector (semantic) search
  • Hybrid search (keyword + vector)
  • RAG: Retrieval Augmented Generation
============================================================
"""

import os
import json
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticSearch,
    SemanticPrioritizedFields,
    SemanticField,
)
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

# Azure AI Search config
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "https://<your-search>.search.windows.net")
SEARCH_KEY      = os.getenv("AZURE_SEARCH_KEY", "<your-search-key>")
INDEX_NAME      = os.getenv("AZURE_SEARCH_INDEX", "ai102-knowledge-base")

# Azure OpenAI config (for embeddings)
OAI_ENDPOINT  = os.getenv("AZURE_OPENAI_ENDPOINT", "https://<your-oai>.openai.azure.com/")
OAI_KEY       = os.getenv("AZURE_OPENAI_KEY", "<your-oai-key>")
OAI_API_VER   = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
EMBED_DEPLOY  = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-ada-002")
CHAT_DEPLOY   = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")

VECTOR_DIMENSIONS = 1536  # text-embedding-ada-002 dimensions

def get_search_index_client():
    return SearchIndexClient(
        endpoint=SEARCH_ENDPOINT,
        credential=AzureKeyCredential(SEARCH_KEY)
    )

def get_search_client():
    return SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(SEARCH_KEY)
    )

def get_openai_client():
    return AzureOpenAI(
        azure_endpoint=OAI_ENDPOINT,
        api_key=OAI_KEY,
        api_version=OAI_API_VER
    )

# ── 1. Create Search Index ────────────────────────────────
def create_search_index() -> None:
    """
    Create an Azure AI Search index with vector search support.
    Includes both keyword fields and a vector field for embeddings.
    """
    index_client = get_search_index_client()

    # Define index fields
    fields = [
        SimpleField(name="id",         type=SearchFieldDataType.String, key=True),
        SearchableField(name="title",  type=SearchFieldDataType.String),
        SearchableField(name="content",type=SearchFieldDataType.String),
        SearchableField(name="category",type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="source",     type=SearchFieldDataType.String),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=VECTOR_DIMENSIONS,
            vector_search_profile_name="hnsw-profile"
        ),
    ]

    # Vector search configuration
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw-algo",
                parameters={
                    "m": 4,                  # Connections per node
                    "efConstruction": 400,   # Build-time accuracy
                    "efSearch": 500,         # Query-time accuracy
                    "metric": "cosine"       # Distance metric
                }
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="hnsw-profile",
                algorithm_configuration_name="hnsw-algo"
            )
        ]
    )

    # Semantic search configuration
    semantic_config = SemanticConfiguration(
        name="semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="title"),
            content_fields=[SemanticField(field_name="content")]
        )
    )
    semantic_search = SemanticSearch(configurations=[semantic_config])

    # Create index
    index = SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search
    )

    result = index_client.create_or_update_index(index)
    print(f"  ✅ Index '{result.name}' created/updated")
    print(f"     Fields: {[f.name for f in result.fields]}")

# ── 2. Generate Embeddings ────────────────────────────────
def embed(text: str) -> list[float]:
    """Generate embedding vector for a text string."""
    oai = get_openai_client()
    response = oai.embeddings.create(model=EMBED_DEPLOY, input=text)
    return response.data[0].embedding

# ── 3. Index Documents ────────────────────────────────────
def index_documents(documents: list[dict]) -> None:
    """
    Upload documents to the search index.
    Each document gets an embedding vector for semantic search.
    """
    client = get_search_client()

    print("\n" + "="*65)
    print("  INDEXING DOCUMENTS")
    print("="*65)

    docs_with_vectors = []
    for doc in documents:
        text_to_embed = f"{doc['title']} {doc['content']}"
        print(f"  Embedding: '{doc['title']}'...")
        vector = embed(text_to_embed)
        doc_with_vector = {**doc, "content_vector": vector}
        docs_with_vectors.append(doc_with_vector)

    result = client.upload_documents(documents=docs_with_vectors)
    success = sum(1 for r in result if r.succeeded)
    print(f"\n  ✅ Indexed {success}/{len(documents)} documents")

# ── 4. Keyword Search ─────────────────────────────────────
def keyword_search(query: str, top: int = 3) -> list[dict]:
    """Traditional full-text keyword search."""
    client = get_search_client()

    results = client.search(
        search_text=query,
        select=["id", "title", "content", "category"],
        top=top
    )

    print("\n" + "="*65)
    print(f"  KEYWORD SEARCH: '{query}'")
    print("="*65)

    found = []
    for result in results:
        print(f"\n  [{result['@search.score']:.2f}] {result['title']}")
        print(f"  {result['content'][:150]}...")
        found.append(result)
    return found

# ── 5. Vector (Semantic) Search ───────────────────────────
def vector_search(query: str, top: int = 3) -> list[dict]:
    """Pure vector similarity search using embeddings."""
    client = get_search_client()

    query_vector = embed(query)

    results = client.search(
        search_text=None,          # No keyword search
        vector_queries=[
            VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top,
                fields="content_vector"
            )
        ],
        select=["id", "title", "content", "category"],
        top=top
    )

    print("\n" + "="*65)
    print(f"  VECTOR SEARCH: '{query}'")
    print("="*65)

    found = []
    for result in results:
        print(f"\n  [{result['@search.score']:.4f}] {result['title']}")
        print(f"  {result['content'][:150]}...")
        found.append(result)
    return found

# ── 6. Hybrid Search ──────────────────────────────────────
def hybrid_search(query: str, top: int = 3) -> list[dict]:
    """
    Combines keyword + vector search for best results.
    RRF (Reciprocal Rank Fusion) merges both result lists.
    """
    client = get_search_client()
    query_vector = embed(query)

    results = client.search(
        search_text=query,         # Keyword component
        vector_queries=[
            VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top,
                fields="content_vector"
            )
        ],
        select=["id", "title", "content", "category"],
        top=top,
        query_type="semantic",
        semantic_configuration_name="semantic-config"
    )

    print("\n" + "="*65)
    print(f"  HYBRID SEARCH: '{query}'")
    print("="*65)

    found = []
    for result in results:
        score = result.get("@search.reranker_score") or result.get("@search.score", 0)
        print(f"\n  [{score:.4f}] {result['title']}")
        print(f"  {result['content'][:150]}...")
        found.append(result)
    return found

# ── 7. RAG Pipeline ───────────────────────────────────────
def rag_query(user_question: str) -> str:
    """
    Full Retrieval Augmented Generation pipeline:
    1. Search for relevant documents
    2. Build context from search results
    3. Send to GPT with context for grounded answer
    """
    oai = get_openai_client()

    print("\n" + "="*65)
    print("  RAG — RETRIEVAL AUGMENTED GENERATION")
    print("="*65)
    print(f"  Question: {user_question}")

    # Step 1: Retrieve
    relevant_docs = vector_search(user_question, top=3)

    # Step 2: Build context
    context = "\n\n".join([
        f"[Source: {doc['title']}]\n{doc['content']}"
        for doc in relevant_docs
    ])

    # Step 3: Generate grounded response
    system_prompt = """You are a helpful AI assistant.
Answer questions using ONLY the provided context.
If the answer is not in the context, say 'I don't have enough information.'
Always cite the source title when referencing information."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_question}"}
    ]

    response = oai.chat.completions.create(
        model=CHAT_DEPLOY,
        messages=messages,
        temperature=0,
        max_tokens=500
    )

    answer = response.choices[0].message.content
    print(f"\n  Grounded Answer:\n  {answer}")
    print(f"\n  Sources used:")
    for doc in relevant_docs:
        print(f"    • {doc['title']}")
    return answer

# ── Sample Documents ───────────────────────────────────────
SAMPLE_DOCUMENTS = [
    {
        "id": "1",
        "title": "Azure AI Language Service Overview",
        "content": "Azure AI Language provides NLP capabilities including sentiment analysis, key phrase extraction, named entity recognition, PII detection, text summarization, and custom text classification.",
        "category": "NLP",
        "source": "Azure Documentation"
    },
    {
        "id": "2",
        "title": "Zero Trust Security Model",
        "content": "Zero Trust is a security model that assumes no user or device is trusted by default. It requires verification for every access request using principles: verify explicitly, use least privilege access, and assume breach.",
        "category": "Security",
        "source": "Security Training"
    },
    {
        "id": "3",
        "title": "Azure OpenAI Service",
        "content": "Azure OpenAI provides REST API access to OpenAI's powerful models including GPT-4o and DALL-E 3. It supports chat completions, embeddings, function calling, and image generation within Azure's enterprise security framework.",
        "category": "AI",
        "source": "Azure Documentation"
    },
    {
        "id": "4",
        "title": "Cybersecurity Threat Categories",
        "content": "Major cybersecurity threats include phishing attacks targeting credentials, ransomware encrypting data for payment, SQL injection exploiting database vulnerabilities, and DDoS attacks overwhelming network resources.",
        "category": "Security",
        "source": "Security Training"
    },
    {
        "id": "5",
        "title": "CLU replaces LUIS",
        "content": "Conversational Language Understanding (CLU) is the successor to LUIS which was retired in September 2025. CLU supports intent detection and entity extraction for conversational AI applications through the Azure Language service.",
        "category": "NLP",
        "source": "Azure Documentation"
    },
]

# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  AZURE AI SEARCH + RAG PIPELINE")

    # Step 1: Create index
    create_search_index()

    # Step 2: Index documents with embeddings
    index_documents(SAMPLE_DOCUMENTS)

    # Step 3: Different search types
    keyword_search("cybersecurity threats")
    vector_search("how does NLP work in Azure")
    hybrid_search("what replaced LUIS for intent detection")

    # Step 4: Full RAG pipeline
    rag_query("What security model should I use for Azure applications?")

    print("\n  KEY POINTS FOR AI-102:")
    print("  • SearchIndexClient → manage indexes")
    print("  • SearchClient → query indexes")
    print("  • vector_search_dimensions must match embedding model (1536 for ada-002)")
    print("  • VectorizedQuery wraps the embedding for search")
    print("  • Hybrid = search_text + vector_queries in same call")
    print("  • Semantic search adds reranker score on top of hybrid")
    print("  • RAG pattern: retrieve → build context → generate")
    print("  • grounding prevents hallucination by anchoring to real data")
    print("="*65 + "\n")
