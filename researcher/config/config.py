import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # LLM settings
    FAST_LLM: str = "openai:gpt-4o-mini"
    SMART_LLM: str = "openai:gpt-4o"
    STRATEGIC_LLM: str = "openai:gpt-4o"
    TEMPERATURE: float = 0.4
    FAST_TOKEN_LIMIT: int = 2000
    SMART_TOKEN_LIMIT: int = 4000
    STRATEGIC_TOKEN_LIMIT: int = 4000
    SUMMARY_TOKEN_LIMIT: int = 700

    # Retriever settings
    RETRIEVER: str = "duckduckgo"
    MAX_SEARCH_RESULTS_PER_QUERY: int = 5

    # Scraper settings
    SCRAPER: str = "bs4"
    MAX_SCRAPER_WORKERS: int = 15
    BROWSE_CHUNK_MAX_LENGTH: int = 8192

    # Embedding and RAG settings
    EMBEDDING: str = "openai:text-embedding-3-small"
    SIMILARITY_THRESHOLD: float = 0.42
    MAX_CONTEXT_CHUNKS: int = 50
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 80

    # Report settings
    TOTAL_WORDS: int = 1200
    REPORT_FORMAT: str = "APA"
    LANGUAGE: str = "english"
    MAX_ITERATIONS: int = 3
    MAX_SUBTOPICS: int = 3
    MAX_SECTIONS: int = 3

    # Source settings
    REPORT_SOURCE: str = "web"
    DOC_PATH: str = "./my-docs"

    def __init__(self, config_file: str | None = None):
        self._apply_defaults()
        self._load_env()
        if config_file:
            self._load_file(config_file)

    def _apply_defaults(self):
        for f in self.__dataclass_fields__.values():
            setattr(self, f.name, f.default)

    def _load_env(self):
        _int = lambda k, d: int(v) if (v := os.getenv(k, "").strip()) else int(d)
        _float = lambda k, d: float(v) if (v := os.getenv(k, "").strip()) else float(d)
        _str = lambda k, d: v if (v := os.getenv(k, "").strip()) else d

        self.FAST_LLM = _str("FAST_LLM", self.FAST_LLM)
        self.SMART_LLM = _str("SMART_LLM", self.SMART_LLM)
        self.STRATEGIC_LLM = _str("STRATEGIC_LLM", self.STRATEGIC_LLM)
        self.TEMPERATURE = _float("TEMPERATURE", str(self.TEMPERATURE))
        self.FAST_TOKEN_LIMIT = _int("FAST_TOKEN_LIMIT", str(self.FAST_TOKEN_LIMIT))
        self.SMART_TOKEN_LIMIT = _int("SMART_TOKEN_LIMIT", str(self.SMART_TOKEN_LIMIT))
        self.STRATEGIC_TOKEN_LIMIT = _int("STRATEGIC_TOKEN_LIMIT", str(self.STRATEGIC_TOKEN_LIMIT))
        self.SUMMARY_TOKEN_LIMIT = _int("SUMMARY_TOKEN_LIMIT", str(self.SUMMARY_TOKEN_LIMIT))

        self.RETRIEVER = _str("RETRIEVER", self.RETRIEVER)
        self.MAX_SEARCH_RESULTS_PER_QUERY = _int("MAX_SEARCH_RESULTS_PER_QUERY", str(self.MAX_SEARCH_RESULTS_PER_QUERY))

        self.SCRAPER = _str("SCRAPER", self.SCRAPER)
        self.MAX_SCRAPER_WORKERS = _int("MAX_SCRAPER_WORKERS", str(self.MAX_SCRAPER_WORKERS))
        self.BROWSE_CHUNK_MAX_LENGTH = _int("BROWSE_CHUNK_MAX_LENGTH", str(self.BROWSE_CHUNK_MAX_LENGTH))

        self.EMBEDDING = _str("EMBEDDING", self.EMBEDDING)
        self.SIMILARITY_THRESHOLD = _float("SIMILARITY_THRESHOLD", str(self.SIMILARITY_THRESHOLD))
        self.MAX_CONTEXT_CHUNKS = _int("MAX_CONTEXT_CHUNKS", str(self.MAX_CONTEXT_CHUNKS))
        self.CHUNK_SIZE = _int("CHUNK_SIZE", str(self.CHUNK_SIZE))
        self.CHUNK_OVERLAP = _int("CHUNK_OVERLAP", str(self.CHUNK_OVERLAP))

        self.TOTAL_WORDS = _int("TOTAL_WORDS", str(self.TOTAL_WORDS))
        self.REPORT_FORMAT = _str("REPORT_FORMAT", self.REPORT_FORMAT)
        self.LANGUAGE = _str("LANGUAGE", self.LANGUAGE)
        self.MAX_ITERATIONS = _int("MAX_ITERATIONS", str(self.MAX_ITERATIONS))
        self.MAX_SUBTOPICS = _int("MAX_SUBTOPICS", str(self.MAX_SUBTOPICS))
        self.MAX_SECTIONS = _int("MAX_SECTIONS", str(self.MAX_SECTIONS))

        self.REPORT_SOURCE = _str("REPORT_SOURCE", self.REPORT_SOURCE)
        self.DOC_PATH = _str("DOC_PATH", self.DOC_PATH)

    def _load_file(self, path: str):
        file = Path(path)
        if not file.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with file.open() as fh:
            data: dict = json.load(fh)
        _fields = {f.name for f in self.__dataclass_fields__.values()}
        for key, value in data.items():
            if key in _fields:
                setattr(self, key, value)

    @staticmethod
    def parse_provider_and_model(llm_string: str) -> tuple[str, str]:
        if ":" not in llm_string:
            raise ValueError(
                f"Expected 'provider:model_name' format, got: {llm_string!r}"
            )
        provider, _, model = llm_string.partition(":")
        return provider.strip(), model.strip()
