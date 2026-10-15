import os
import asyncio
from functools import partial
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache,openai_embed
from lightrag.utils import EmbeddingFunc
import numpy as np

#########
# Uncomment the below two lines if running in a jupyter notebook to handle the async nature of rag.insert()
# import nest_asyncio
# nest_asyncio.apply()
#########
WORKING_DIR = "./mongodb_test_dir"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)

os.environ["MONGO_URI"] = "mongodb://127.0.0.1:27017/?directConnection=true"
os.environ["MONGO_DATABASE"] = "graph"
os.environ["MONGO_KG_COLLECTION"] = "MDB_KG"

# Embedding Configuration and Functions
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_MAX_TOKEN_SIZE = int(os.environ.get("EMBEDDING_MAX_TOKEN_SIZE", 8192))


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


async def create_embedding_function_instance():
    # Create embedding function instance
    embedding_func=EmbeddingFunc(
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
        max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
        func=partial(
            openai_embed.func,  # Access the unwrapped function to avoid double EmbeddingFunc wrapping
            model=os.getenv("EMBEDDING_MODEL", "bge-m3:latest"),
            base_url=os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434"),
            api_key=os.getenv("EMBEDDING_BINDING_API_KEY", ""),
        ),
    )
    return embedding_func

async def initialize_rag():
    embedding_func_instance = await create_embedding_function_instance()

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func_instance,
        graph_storage="MongoGraphStorage",
        log_level="DEBUG",
    )

    await rag.initialize_storages()  # Auto-initializes pipeline_status
    return rag


def main():
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())

    with open("./book.txt", "r", encoding="utf-8") as f:
        rag.insert(f.read())

    # Perform naive search
    print(
        rag.query(
            "What are the top themes in this story?", param=QueryParam(mode="naive")
        )
    )

    # Perform local search
    print(
        rag.query(
            "What are the top themes in this story?", param=QueryParam(mode="local")
        )
    )

    # Perform global search
    print(
        rag.query(
            "What are the top themes in this story?", param=QueryParam(mode="global")
        )
    )

    # Perform hybrid search
    print(
        rag.query(
            "What are the top themes in this story?", param=QueryParam(mode="hybrid")
        )
    )


if __name__ == "__main__":
    main()
