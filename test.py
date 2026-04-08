# test_local_models.py
from __future__ import annotations

from pathlib import Path

import spacy
import tiktoken
from sentence_transformers import SentenceTransformer

SPACY_MODEL_PATH = (
    r"D:\Code\LLM_IE\.venv\Lib\site-packages\en_core_web_sm\en_core_web_sm-3.8.0"
)
TOKENIZER_NAME = "cl100k_base"
EMBEDDING_MODEL_PATH = r"C:\Users\zhulin\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\c9745ed1d9f207416be6d2e6f8de32d1f16199bf"


def test_spacy_model() -> None:
    print("\n" + "=" * 80)
    print("[1] Testing spaCy local model")
    print("=" * 80)

    model_path = Path(SPACY_MODEL_PATH)
    print(f"Configured path: {model_path}")
    print(f"Path exists: {model_path.exists()}")

    if not model_path.exists():
        raise FileNotFoundError(f"spaCy model path does not exist: {model_path}")

    nlp = spacy.load(str(model_path))
    doc = nlp("This is a test sentence for spaCy.")
    print("Load success.")
    print(f"Resolved nlp.path: {nlp.path}")
    print(f"Tokens: {[token.text for token in doc]}")
    print(f"Sentences: {[sent.text for sent in doc.sents]}")


def test_tiktoken() -> None:
    print("\n" + "=" * 80)
    print("[2] Testing tiktoken local availability")
    print("=" * 80)

    enc = tiktoken.get_encoding(TOKENIZER_NAME)
    text = "This is a test sentence for cl100k_base."
    token_ids = enc.encode(text)

    print("Load success.")
    print(f"Encoding name: {enc.name}")
    print(f"Input text: {text}")
    print(f"Token ids: {token_ids}")
    print(f"Token count: {len(token_ids)}")


def test_embedding_model() -> None:
    print("\n" + "=" * 80)
    print("[3] Testing SentenceTransformer local model")
    print("=" * 80)

    model_path = Path(EMBEDDING_MODEL_PATH)
    print(f"Configured path: {model_path}")
    print(f"Path exists: {model_path.exists()}")

    if not model_path.exists():
        raise FileNotFoundError(f"Embedding model path does not exist: {model_path}")

    model = SentenceTransformer(str(model_path))
    sentences = [
        "Wheat yield increased under improved irrigation.",
        "Rice production declined because of drought.",
    ]
    embeddings = model.encode(sentences, normalize_embeddings=True)

    print("Load success.")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"First sentence: {sentences[0]}")
    print(f"Second sentence: {sentences[1]}")
    print(f"Vector dim: {len(embeddings[0])}")


def main() -> None:
    print("Start testing local model access...")

    try:
        test_spacy_model()
    except Exception as e:
        print(f"[FAILED] spaCy test failed: {e}")

    try:
        test_tiktoken()
    except Exception as e:
        print(f"[FAILED] tiktoken test failed: {e}")

    try:
        test_embedding_model()
    except Exception as e:
        print(f"[FAILED] SentenceTransformer test failed: {e}")

    print("\nAll tests finished.")


if __name__ == "__main__":
    main()
