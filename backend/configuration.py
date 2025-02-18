import os
from functools import cache, cached_property
from pathlib import Path
from typing import Any

from langchain_community.cache import SQLiteCache
from langchain_core.embeddings import Embeddings
from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai.chat_models.base import BaseChatOpenAI
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
    langsmith_api_key: str = Field("", description="LangSmith API Key")
    langsmith_project: str = Field(
        "", description="Project name in LangSmith. If empty disable tracing"
    )

    # Parameters
    chunk_size: int = Field(500, description="Chunk size for data processing")
    chunk_overlap: int = Field(
        20, description="Number of overlapping characters between chunks"
    )
    similarity_top_k: int = Field(
        4, description="Retriever's final number of returned nodes"
    )

    # Input for development purposes
    cv_path: Path = Field(..., description="Full path to CV")
    job_description: str = Field(..., description="Job description")

    @model_validator(mode="before")
    @classmethod
    def validate_cv_path(cls, data: Any) -> Any:
        cv_path = data.get("cv_path", None)
        if cv_path is not None:
            cv_path = Path(cv_path)
        else:
            cv_path = Path(input("Type the full path of the resume"))
            data["cv_path"] = cv_path
        if not cv_path.exists():
            raise ValueError(f"{cv_path}: CV does not exist")

        return data

    @model_validator(mode="before")
    @classmethod
    def validate_job_description(cls, data: Any) -> Any:
        job_description = data.get("job_description", "")
        if job_description == "":
            data["job_description"] = input("Type the job description")

        return data

    @property
    def is_tracing(self) -> bool:
        if self.langsmith_api_key == "":
            if self.langsmith_project == "":
                return False
            else:
                raise ValueError("Project at LangSmith is set but the api key is empty")
        else:
            if self.langsmith_project == "":
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

    @model_validator(mode="after")
    def set_tracing_env_vars(self):
        if self.is_tracing:
            os.environ["LANGSMITH_API_KEY"] = self.langsmith_api_key
            os.environ["LANGSMITH_TRACING"] = "true"

        return self


def config_cache(on: bool = True) -> None:
    if on:
        set_llm_cache(SQLiteCache(database_path=".SQLiteCache_analysis.db"))
    else:
        set_llm_cache(None)
