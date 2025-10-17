import numpy as np


def search(query, embedder, index, docs, top_k=3):
    q_emb = embedder.encode([query])
    _, Index = index.search(np.array(q_emb), top_k)
    return [docs[i] for i in Index[0]]
