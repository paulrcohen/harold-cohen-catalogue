#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 12:06:02 2025

@author: prcohen
"""

#!/usr/bin/env python3
"""
Modular Harold Cohen Catalogue App
Clean separation of concerns with cloud-ready persistence
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

# Import our modules
from cloud_search import get_search_engine

# Page configuration
st.set_page_config(
    page_title="Harold Cohen Catalogue",
    page_icon="🎨",
    layout="wide"
)


# ==================== UI COMPONENTS ====================

def render_header():
    """Render the main header and status"""
    st.title("🎨 Harold Cohen Catalogue Raisonné")
    st.markdown("*Comprehensive search system for Harold Cohen's work and correspondence*")
    
    # Status bar
    engine = get_search_engine()
    stats = engine.get_stats()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Documents", stats.get('document_count', 0))
    with col2:
        if stats.get('backup_exists', False):
            st.success("💾 Data Backed Up")
        else:
            st.info("💾 No Backup Yet")
    with col3:
        st.metric("Status", stats.get('status', 'unknown').title())


def render_search_interface():
    """Render the search interface"""
    st.header("🔍 Search Collection")
    
    engine = get_search_engine()
    
    # Search input
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., Brooklyn museum, National Theatre painting, shipping delays, AARON program..."
        )
    with col2:
        num_results = st.selectbox("Max results:", [3, 5, 10], index=1)
    
    # Search button and results
    if st.button("🔍 Search", type="primary") and query:
        with st.spinner("Searching..."):
            results = engine.search(query, n_results=num_results)
        
        display_search_results(results, query)


def display_search_results(results, query):
    """Display search results"""
    if results:
        st.subheader(f"📄 Found {len(results)} results for '{query}'")
        
        for i, result in enumerate(results, 1):
            with st.expander(f"Result {i} - {result['metadata'].get('source', 'Unknown')} (Relevance: {result['relevance']:.2f})"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write("**Content:**")
                    content = result['content']
                    if len(content) > 500:
                        st.write(content[:500] + "...")
                        with st.expander("Show full content"):
                            st.write(content)
                    else:
                        st.write(content)
                
                with col2:
                    st.write("**Metadata:**")
                    metadata = result['metadata']
                    for key, value in metadata.items():
                        if key in ['source', 'type', 'date', 'sender', 'subject']:
                            st.write(f"**{key}:** {value}")
    else:
        st.warning("No results found. Try different search terms or check if documents are loaded.")


def render_upload_interface():
    """Render the file upload interface"""
    st.header("📤 Add Documents")
    
    engine = get_search_engine()
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with documents/emails",
        type=['csv'],
        help="CSV should contain a column with text content (emails, documents, etc.)"
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded CSV with {len(df)} rows")
            
            # Preview
            with st.expander("📋 Preview Data"):
                st.dataframe(df.head())
                st.write(f"**Columns:** {', '.join(df.columns)}")
            
            # Column selection
            col1, col2 = st.columns(2)
            
            with col1:
                # Auto-detect text column
                text_candidates = [col for col in df.columns 
                                 if any(keyword in col.lower() 
                                       for keyword in ['text', 'message', 'content', 'body', 'email'])]
                
                default_col = text_candidates[0] if text_candidates else df.columns[0]
                
                text_column = st.selectbox(
                    "Select text column:",
                    df.columns.tolist(),
                    index=df.columns.tolist().index(default_col)
                )
            
            with col2:
                source_name = st.text_input(
                    "Collection name:",
                    value=uploaded_file.name.replace('.csv', ''),
                    help="Name for this collection of documents"
                )
            
            # Preview selected column
            if text_column:
                st.write("**Sample from selected column:**")
                sample_text = str(df[text_column].iloc[0])
                st.write(sample_text[:300] + "..." if len(sample_text) > 300 else sample_text)
            
            # Ingest button
            if st.button("📥 Add to Collection", type="primary"):
                with st.spinner("Processing documents..."):
                    result = engine.ingest_csv(df, text_column, source_name)
                
                if result['status'] == 'success':
                    st.success(f"✅ Successfully added {result['documents_added']} documents!")
                    st.info(f"Total documents in collection: {result['total_documents']}")
                    st.rerun()
                else:
                    st.error(f"❌ Upload failed: {result['message']}")
                    
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")


def render_collection_overview():
    """Render collection overview and management"""
    st.header("📊 Collection Overview")
    
    engine = get_search_engine()
    stats = engine.get_stats()
    
    # Summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Documents", stats.get('document_count', 0))
    with col2:
        backup_status = "✅ Active" if stats.get('backup_exists') else "❌ None"
        st.metric("Backup Status", backup_status)
    with col3:
        st.metric("Collection Status", stats.get('status', 'Unknown').title())
    
    # Collection details
    if stats.get('document_count', 0) > 0:
        st.subheader("📚 Sample Documents")
        
        # Show a few sample documents
        sample_results = engine.search("Harold Cohen", n_results=3)
        if sample_results:
            for i, result in enumerate(sample_results, 1):
                with st.expander(f"Sample {i} - {result['metadata'].get('source', 'Unknown')}"):
                    st.write(result['content'][:200] + "...")
                    st.write(f"**Source:** {result['metadata'].get('source', 'Unknown')}")
                    st.write(f"**Type:** {result['metadata'].get('type', 'Unknown')}")
    
    # Management actions
    st.subheader("🔧 Collection Management")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Force Backup Save"):
            engine._save_backup()
            st.success("Backup saved!")
    
    with col2:
        if st.button("📋 Export Statistics"):
            st.json(stats)


# ==================== MAIN APP ====================

def main():
    """Main application"""
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'search'
    
    # Render header
    render_header()
    
    # Navigation
    tab1, tab2, tab3 = st.tabs(["🔍 Search", "📤 Upload", "📊 Overview"])
    
    with tab1:
        render_search_interface()
    
    with tab2:
        render_upload_interface()
    
    with tab3:
        render_collection_overview()


if __name__ == "__main__":
    main()
