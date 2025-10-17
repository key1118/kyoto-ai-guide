import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_answer(query, retriever):
    context = "\n".join(retriever(query))
    prompt = f"""
あなたは京都の観光ガイドです。
以下の情報を参考に、京都弁で自然に答えてください。

【参考情報】
{context}

【質問】
{query}

【回答】
"""
    res = client.chat.completions.create(
        model="ft:gpt-4o-mini-2024-07-18:keitaro::CRXr0O9Q",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.7,
    )
    return res.choices[0].message.content.strip()
