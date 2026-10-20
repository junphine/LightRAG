import os
import asyncio
from functools import partial
from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete
from lightrag.llm.openai import openai_complete_if_cache,openai_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
#########
# Uncomment the below two lines if running in a jupyter notebook to handle the async nature of rag.insert()
# import nest_asyncio
# nest_asyncio.apply()
#########

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)

WORKING_DIR = "./custom_kg"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

os.environ['LIGHTRAG_GRAPH_STORAGE'] = 'GremlinStorage'

async def llm_model_func(
        prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        os.getenv("LLM_MODEL", "deepseek-chat"),
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_BINDING_HOST", "https://api.deepseek.com"),
        **kwargs,
    )


rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,  # Use gpt_4o_mini_complete LLM model
    kv_storage=os.getenv("LIGHTRAG_KV_STORAGE", "JsonKVStorage"),
    graph_storage=os.getenv("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage"),
    doc_status_storage=os.getenv(
        "LIGHTRAG_DOC_STATUS_STORAGE", "JsonDocStatusStorage"
    ),
    vector_storage=os.getenv(
        "LIGHTRAG_VECTOR_STORAGE", "NoopVectorDBStorage"
    ),
    # llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
    embedding_func=EmbeddingFunc(
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
        max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
        func=partial(
            openai_embed.func,  # Access the unwrapped function to avoid double EmbeddingFunc wrapping
            model=os.getenv("EMBEDDING_MODEL", "bge-m3:latest"),
            base_url=os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434"),
            api_key=os.getenv("EMBEDDING_BINDING_API_KEY", ""),
        ),
    ),
)

custom_kg = {
    "entities": [
        {
            "entity_name": "CompanyA",
            "entity_type": "Organization",
            "description": "A major technology company",
            "source_id": "Source1",
        },
        {
            "entity_name": "ProductX",
            "entity_type": "Product",
            "description": "A popular product developed by CompanyA",
            "source_id": "Source1",
        },
        {
            "entity_name": "PersonA",
            "entity_type": "Person",
            "description": "A renowned researcher in AI",
            "source_id": "Source2",
        },
        {
            "entity_name": "UniversityB",
            "entity_type": "Organization",
            "description": "A leading university specializing in technology and sciences",
            "source_id": "Source2",
        },
        {
            "entity_name": "CityC",
            "entity_type": "Location",
            "description": "A large metropolitan city known for its culture and economy",
            "source_id": "Source3",
        },
        {
            "entity_name": "EventY",
            "entity_type": "Event",
            "description": "An annual technology conference held in CityC",
            "source_id": "Source3",
        },
    ],
    "relationships": [
        # Each real source_id contributes an evidence-count floor of 1.0.
        # Larger values are optional boosts; omit source_id for a source-less
        # non-negative fractional weight.
        {
            "src_id": "CompanyA",
            "tgt_id": "ProductX",
            "description": "CompanyA develops ProductX",
            "keywords": "develop, produce",
            "weight": 1.0,
            "source_id": "Source1",
        },
        {
            "src_id": "PersonA",
            "tgt_id": "UniversityB",
            "description": "PersonA works at UniversityB",
            "keywords": "employment, affiliation",
            "weight": 1.25,
            "source_id": "Source2",
        },
        {
            "src_id": "CityC",
            "tgt_id": "EventY",
            "description": "EventY is hosted in CityC",
            "keywords": "host, location",
            "weight": 1.0,
            "source_id": "Source3",
        },
    ],
    "chunks": [
        {
            "content": "ProductX, developed by CompanyA, has revolutionized the market with its cutting-edge features.",
            "source_id": "Source1",
            "source_chunk_index": 0,
        },
        {
            "content": "One outstanding feature of ProductX is its advanced AI capabilities.",
            "source_id": "Source1",
            "source_chunk_index": 1,
        },
        {
            "content": "PersonA is a prominent researcher at UniversityB, focusing on artificial intelligence and machine learning.",
            "source_id": "Source2",
            "source_chunk_index": 0,
        },
        {
            "content": "EventY, held in CityC, attracts technology enthusiasts and companies from around the globe.",
            "source_id": "Source3",
            "source_chunk_index": 0,
        },
        {
            "content": "None",
            "source_id": "UNKNOWN",
            "source_chunk_index": 0,
        },
    ],
}

async def main() -> None:
    await rag.initialize_storages()
    await rag.ainsert_custom_kg(custom_kg)

if __name__ == "__main__":
    asyncio.run(main())
