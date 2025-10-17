import streamlit as st
from src.rag_pipeline import setup_rag
from src.llm_client import generate_answer
st.set_page_config(page_title="京都AI観光ガイドくん", page_icon="🪷")
st.title("🪷 京都AI観光ガイド")
st.write("RAG + Fine-tuned GPT-4o-mini")

retriever = setup_rag()

query = st.text_input("京都について聞いてみましょう！", placeholder="例：金閣寺の歴史を教えて")
if query:
    with st.spinner("考え中..."):
        answer = generate_answer(query, retriever)
        st.success(answer)
