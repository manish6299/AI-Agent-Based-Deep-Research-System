import streamlit as st
from graph import build_graph

st.set_page_config(page_title="Deep Research AI Agent", layout="centered")
st.title("🔍 Dual-Agent Deep Research System")

query = st.text_input("Enter your research query:")
submit = st.button("Run Deep Research")

if submit and query:
    with st.spinner("Researching..."):
        try:
            # Build the graph
            graph = build_graph()
            
            # Invoke the graph with the proper initial state
            result = graph.invoke({"query": query})
            
            st.subheader("📄 Answer")
            st.write(result["answer"])
            
        except Exception as e:
            st.error(f"Research failed: {str(e)}")