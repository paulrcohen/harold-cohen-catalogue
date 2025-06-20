#!/usr/bin/env python3
"""
Fresh start - absolute minimal working version
"""

import streamlit as st
import chromadb

st.set_page_config(page_title="Harold Cohen - Fresh Start", page_icon="🎨")

def main():
    st.title("🎨 Harold Cohen - Fresh Start")
    
    # Test 1: Basic Streamlit
    st.success("✅ Streamlit is working!")
    
    # Test 2: ChromaDB
    try:
        client = chromadb.Client()  # In-memory only for now
        collection = client.create_collection("test")
        
        # Add test data
        collection.add(
            documents=["Harold Cohen was a computer artist", "He created the AARON program"],
            ids=["doc1", "doc2"]
        )
        
        st.success("✅ ChromaDB is working!")
        st.write(f"Collection has {collection.count()} documents")
        
        # Test search
        query = st.text_input("Test search:", "Harold Cohen")
        if query:
            results = collection.query(query_texts=[query], n_results=2)
            st.write("Results:", results['documents'][0])
            
    except Exception as e:
        st.error(f"❌ ChromaDB failed: {e}")

if __name__ == "__main__":
    main()
