from sentence_transformers import SentenceTransformer
from transformers import pipeline

# 埋め込みモデルの確認
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("✅ Embedding model loaded")

# 軽量LLMの確認（Llama系 or Mistral系）
llm = pipeline("text-generation", model="microsoft/phi-2", device_map="auto")
print("✅ LLM pipeline ready")
