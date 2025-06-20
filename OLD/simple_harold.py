#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 11:20:55 2025

@author: prcohen
"""

#!/usr/bin/env python3
"""
Simple Harold Cohen Search App - Building up from working foundation
"""

import streamlit as st
import chromadb
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Harold Cohen Search", page_icon="🎨", layout="wide")

# Initialize ChromaDB (in-memory for simplicity)
@st.cache_resource
def get_chromadb():
    client = chromadb.Client()
    collection = client.get_or_create_collection("harold_cohen")
    
    # Add some initial test data if empty
    if collection.count() == 0:
        test_docs = [
            "Harold Cohen was a pioneering computer artist who created the AARON drawing program from the 1970s onwards.",
            "The Brooklyn Museum exhibition featured Harold Cohen's computer-generated artwork and large-scale prints.",
            "There were shipping delays with the international exhibition materials due to customs paperwork.",
            "The National Theatre painting was a significant 6x4 foot work from Cohen's figurative period.",
            "Installation issues occurred with the National Theater painting - there were problems with the frame."
        ]
        
        test_metadata = [
            {"source": "biography", "type": "reference"},
            {"source": "brooklyn_exhibition", "type": "email"},
            {"source": "shipping_update", "type": "email"},
            {"source": "artwork_record", "type": "catalog"},
            {"source": "installation_notes", "type": "email"}
        ]
        
        collection.add(
            documents=test_docs,
            metadatas=test_metadata,
            ids=[f"doc_{i}" for i in range(len(test_docs))]
        )
    
    return collection

def main():
    st.title("🎨 Harold Cohen Catalogue Search")
    st.markdown("*Simple search system for Harold Cohen's work and correspondence*")
    
    # Get collection
    collection = get_chromadb()
    
    # Show status
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Documents in Collection", collection.count())
    with col2:
        st.info("💾 In-Memory Storage (data resets on restart)")
    
    # Search interface
    st.header("🔍 Search")
    
    query = st.text_input(
        "Enter your search query:",
        placeholder="e.g., Brooklyn museum, National Theatre painting, shipping delays..."
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_btn = st.button("Search", type="primary")
    with col2:
        num_results = st.selectbox("Max results:", [3, 5, 10], index=0)
    
    if search_btn and query:
        # Perform search
        try:
            results = collection.query(
                query_texts=[query],
                n_results=num_results
            )
            
            if results['documents'][0]:
                st.subheader(f"📄 Found {len(results['documents'][0])} results")
                
                # Display results
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                ), 1):
                    
                    with st.expander(f"Result {i} - {metadata.get('source', 'Unknown')}"):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write("**Content:**")
                            st.write(doc)
                        
                        with col2:
                            st.write("**Metadata:**")
                            for key, value in metadata.items():
                                st.write(f"**{key}:** {value}")
                            
                            relevance = 1 - distance
                            st.write(f"**Relevance:** {relevance:.2f}")
            else:
                st.warning("No results found. Try different search terms.")
                
        except Exception as e:
            st.error(f"Search failed: {e}")
    
    # File upload section
    st.header("📤 Add More Data")
    
    uploaded_file = st.file_uploader(
        "Upload a CSV file with text content",
        type=['csv'],
        help="CSV should have columns like 'text', 'message', or 'content'"
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded CSV with {len(df)} rows")
            
            # Show preview
            st.write("**Preview:**")
            st.dataframe(df.head(3))
            
            # Select text column
            text_col = st.selectbox(
                "Which column contains the text to search?",
                df.columns.tolist()
            )
            
            if st.button("Add to Collection"):
                # Add documents to collection
                docs_added = 0
                for idx, row in df.iterrows():
                    text = str(row.get(text_col, "")).strip()
                    if text and text != "nan":
                        collection.add(
                            documents=[text],
                            metadatas=[{
                                "source": uploaded_file.name,
                                "row": idx,
                                "type": "uploaded"
                            }],
                            ids=[f"upload_{uploaded_file.name}_{idx}"]
                        )
                        docs_added += 1
                
                st.success(f"✅ Added {docs_added} documents to collection!")
                st.rerun()
                
        except Exception as e:
            st.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
