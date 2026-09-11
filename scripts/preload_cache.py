import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semantic_cs.bootstrap import create_store, preload_hot_faqs
from semantic_cs.text import EmbeddingModel


def main() -> None:
    embedding_model = EmbeddingModel()
    store = create_store()
    count = preload_hot_faqs(store, embedding_model)
    print(f"preloaded_hot_faqs={count}")


if __name__ == "__main__":
    main()
