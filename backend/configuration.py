import os
from functools import cache, cached_property
from pathlib import Path
from typing import Any

from langchain_community.cache import SQLiteCache
from langchain_core.embeddings import Embeddings
from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai.chat_models.base import BaseChatOpenAI
from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PATH_CHUNKS = Path("chunks")
DB_INDEX = Path("db_index")


@cache
class Configuration(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Models
    llm_openai_model: str = Field("gpt-4o", description="LLM model from OpenAI")
    embeddings_openai_model: str = Field(
        "text-embedding-3-small",
        description="Embeddings model from OpenAI",
    )

    # API keys
    openai_api_key: str = Field("", description="OpenAI api key")
    arize_api_key: str = Field(
        "", description="API Key with hosting at Phoenix Arize AI"
    )
    arize_project: str = Field(
        "", description="Project name in Phoenix Arize AI. If empty disable tracing"
    )

    # Parameters
    chunk_size: int = Field(500, description="Chunk size for data processing")
    chunk_overlap: int = Field(
        20, description="Number of overlapping characters between chunks"
    )
    similarity_top_k: int = Field(
        5, description="Retriever's final number of returned nodes"
    )

    # Input for development purposes
    cv_path: Path = Field(..., description="Full path to CV")
    job_description: str = Field(..., description="Job description")

    @model_validator(mode="before")
    @classmethod
    def validate_cv_path(cls, data: Any) -> Any:
        cv_path = Path(data.get("cv_path", None))
        if cv_path is not None and not cv_path.exists():
            raise ValueError(f"{cv_path}: CV does not exist")

        return data

    @property
    def is_tracing(self) -> bool:
        if self.arize_api_key == "":
            if self.arize_project == "":
                return False
            else:
                raise ValueError(
                    "Project at Phoenix Arize AI is set but the api key is empty"
                )
        else:
            if self.arize_project == "":
                return False
            else:
                return True

    @cached_property
    def llm(self) -> BaseChatOpenAI:
        return ChatOpenAI(model_name=self.llm_openai_model, api_key=self.openai_api_key)

    @cached_property
    def embeddings(self) -> Embeddings:
        return OpenAIEmbeddings(
            model=self.embeddings_openai_model, api_key=self.openai_api_key
        )


def config_cache():
    set_llm_cache(SQLiteCache(database_path=".SQLiteCache_analysis.db"))


def config_tracing(config: Configuration) -> None:
    if config.is_tracing:
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = config.arize_api_key
        os.environ["PHOENIX_CLIENT_HEADERS"] = config.arize_api_key
        os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"

        tracer_provider = register(project_name=config.arize_project)

        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
