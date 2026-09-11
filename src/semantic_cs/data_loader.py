from __future__ import annotations
import json
from pathlib import Path
from typing import TypeVar

from semantic_cs.config import DATA_DIR
from semantic_cs.models import FAQItem, KnowledgeDoc, ModelMixin, Product


T = TypeVar("T", bound=ModelMixin)


def _load_model_list(path: Path, model: type[T]) -> list[T]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    return [model.model_validate(item) for item in payload]


def load_faqs() -> list[FAQItem]:
    return _load_model_list(DATA_DIR / "faq.json", FAQItem)


def load_docs() -> list[KnowledgeDoc]:
    return _load_model_list(DATA_DIR / "knowledge_base.json", KnowledgeDoc)


def load_products() -> list[Product]:
    return _load_model_list(DATA_DIR / "products.json", Product)


def load_eval_requests() -> list[dict[str, str]]:
    with (DATA_DIR / "eval_requests.json").open("r", encoding="utf-8") as fp:
        return json.load(fp)

