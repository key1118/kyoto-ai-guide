from src.data_loader import load_documents
from src.embedder import create_embeddings
from src.retriever import search


def setup_rag():
    docs = load_documents("data")
    embedder, index = create_embeddings(docs)

    def retriever(query: str):
        """ユーザーの質問に最も関連するドキュメントを検索"""
        return search(query, embedder, index, docs)

    return retriever
