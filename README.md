# 京都AI観光ガイドくん  
RAG + OpenAI Fine-tuning による京都特化AI観光ガイド

---

## プロジェクト概要

「京都AI観光ガイドくん」は、京都に関する質問に自然な日本語（京都弁）で答えてくれるAIアプリケーションです。  
Retrieval-Augmented Generation (RAG) により、観光ドキュメントから文脈を検索し、  
さらに OpenAI GPT-4o-mini をファインチューニングして京都特化の回答を生成します。

このプロジェクトでは以下を実現しています：

- **RAG構成**：FAISS + SentenceTransformer による関連ドキュメント検索  
- **ファインチューニング**：GPT-4o-mini を京都弁の対話データで再学習  
- **Web UI**：Streamlit による直感的なチャット画面  

---

## アーキテクチャ構成

```text
ユーザー入力
    ↓
SentenceTransformer でベクトル化
    ↓
FAISS によるドキュメント検索
    ↓
OpenAI Fine-tuned GPT-4o-mini で回答生成(入力: プロンプト + 関連性の高いドキュメント2つ)
    ↓
Streamlit UI に出力

---

## ディレクトリ構成

kyoto-ai-guide/
├── app/
│   └── streamlit_app.py          # Webアプリ本体(streamlit)
│
├── src/
│   ├── data_loader.py            # ドキュメント読み込み
│   ├── embedder.py               # ベクトル化・FAISS構築
│   ├── retriever.py              # 検索ロジック
│   ├── llm_client.py             # OpenAIモデル呼び出し
│   └── rag_pipeline.py           # RAG統合セットアップ
│
├── data/                         # 京都観光テキスト群（RAG用）
├── fine_tune/                    # Fine-tuningデータ
├── models/                       # モデルや埋め込みキャッシュ（任意）
├── .env                          # OPENAI_API_KEY（非公開）
└── README.md

---

## 使用技術
| 分類      | 使用技術                           |
| ------- | ------------------------------ |
| 言語モデル   | OpenAI GPT-4o-mini（Fine-tuned） |
| 検索基盤    | FAISS, SentenceTransformer     |
| フロントエンド | Streamlit                      |
| 言語      | Python 3.12                    |
| 環境      | CPU対応（WSL / macOS / Linux）     |

---

## 学びと工夫点

日本語モデルの比較実験：最初はOpenCALM・rinna・phi-2を検証、しかし回答精度と処理速度のトレードオフが問題となり、OpenAIモデルに移行。
GPUを使わず軽量RAG構成を実現：CPUでも高速応答可能。
Fine-tuningの実践：OpenAI公式CLIでデータ整形から学習完了まで一貫実装。
京都弁対応：システムプロンプトと学習データ設計で自然な文体を再現。

---

## 今後の展望

1. 現状RAG用のドキュメントは手作りかつ少量。そのため京都に関するデータを大量に収集できるようなスクレイピングシステムを構築したい
2. GPUを用いればローカルLLMでもおそらく実装可能。コストを下げるのであれば、Google colabなどを活用してOpenAIからローカルLLMへ変更。

---

## 制作者

Keitaro Komatsu (@komatsu.k)
Python / AI開発 / LLM Fine-tuning, GAG
Email: kin29sasuke@outlook.com
GitHub: github.com/key1118


## Web上(OpenAI使用のため現在はローカル環境でのみ公開)

