from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
import glob
import os
from dotenv import load_dotenv
from openai import OpenAI

# === 0. 環境変数の読み込み ===
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY が見つかりません。.env ファイルを確認してください。")

client = OpenAI(api_key=api_key)

# === 1. データ読み込み ===
def load_documents(folder_path="data"):
    docs = []
    for path in glob.glob(os.path.join(folder_path, "*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            docs.append(text)
    return docs

# === 2. ベクトル化 ===
def create_embeddings(docs):
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(docs, convert_to_numpy=True)
    return model, np.array(embeddings)

# === 3. 検索関数 ===
def search(query, embedder, index, docs, top_k=2):
    q_emb = embedder.encode([query])
    D, I = index.search(np.array(q_emb), top_k)
    return [docs[i] for i in I[0]]

# === 4. LLM（LlamaやPhi）で回答生成 ===
def answer(query, retriever):
    context = "\n".join(retriever(query))
    prompt = f"""
あなたは京都の観光ガイドです。
次の情報を参考に、質問に簡潔で自然な日本語で答えてください。
クイズのような形式ではなく、普通の説明文で答えてください。
不明な場合は「分かりません」と答えてください。

【参考情報】
{context}

【質問】
{query}

【回答】
    """
    res = client.chat.completions.create(
        model="ft:gpt-4o-mini-2024-07-18:keitaro::CRXr0O9Q",  # ファインチューニングモデル
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=100
    )
    return res.choices[0].message.content.strip()

# === メイン処理 ===
if __name__ == "__main__":
    print("🔹 ドキュメントを読み込み中...")
    docs = load_documents("data")

    print("🔹 埋め込み生成中...")
    embedder, embeddings = create_embeddings(docs)

    print("🔹 FAISSインデックス作成中...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    retriever = lambda q: search(q, embedder, index, docs)


    print("✅ セットアップ完了！（自作ファインチューニングモデル 使用中）")

    while True:
        query = input("\n🗣 質問をどうぞ（終了するにはexit）: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = answer(query, retriever)
        print("\n🤖 回答:\n", response)
