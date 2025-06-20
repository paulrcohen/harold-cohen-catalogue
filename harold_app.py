#!/usr/bin/env python3
"""
Modular Harold Cohen Catalogue App
Clean separation of concerns with cloud-ready persistence
"""

from typing import List, Dict, Any, Optional
import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

# Import our modules
from simple_vector_search import get_search_engine
from ai_assistant import get_ai_assistant

def check_password():
    """Returns True if password is correct"""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == "hct2025":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password
        st.title("🎨 Harold Cohen Email Archive")
        st.markdown("### Please enter the password to access the research tool")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.markdown("---")
        st.markdown("*This archive contains confidential research materials*")
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input + error
        st.title("🎨 Harold Cohen Email Archive") 
        st.markdown("### Please enter the password to access the research tool")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("❌ Password incorrect - please try again")
        st.markdown("---")
        st.markdown("*This archive contains confidential research materials*")
        return False
    else:
        # Password correct
        return True

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
    ai = get_ai_assistant()
    stats = engine.get_stats()
    ai_stats = ai.get_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Documents", stats.get('document_count', 0))
    with col2:
        if stats.get('backup_exists', False):
            st.success("💾 Data Backed Up")
        else:
            st.info("💾 No Backup Yet")
    with col3:
        if ai_stats.get('available', False):
            st.success("🤖 AI Ready")
        else:
            st.warning("🤖 AI Unavailable")
            # Temporary debug info
            with st.expander("🔍 Debug AI Status"):
                try:
                    # Check if secrets are accessible
                    has_key = "ANTHROPIC_API_KEY" in st.secrets
                    st.write(f"Key in secrets: {has_key}")
                    
                    if has_key:
                        key_preview = st.secrets["ANTHROPIC_API_KEY"][:10] + "..."
                        st.write(f"Key preview: {key_preview}")
                    
                    # Check anthropic import
                    try:
                        import anthropic
                        st.write("Anthropic imported: ✅")
                    except ImportError as e:
                        st.write(f"Anthropic import error: {e}")
                    
                except Exception as e:
                    st.write(f"Debug error: {e}")
    with col4:
        cost = ai_stats.get('total_cost_estimate', 0)
        st.metric("AI Cost", f"${cost:.3f}")


def render_search_interface():
    """Render the search interface"""
    st.header("🔍 Search Collection")
    
    engine = get_search_engine()
    
    # Search input
    col1, col2 = st.columns([3, 1])
    with col1:
        # Check for suggested query
        default_query = st.session_state.get('suggested_query', '')
        if default_query:
            del st.session_state.suggested_query  # Use it once
        
        query = st.text_input(
            "Enter your search query:",
            value=default_query,
            placeholder="e.g., Brooklyn museum, National Theatre painting, shipping delays, AARON program..."
        )
        st.caption("💡 Ignore the 'Press Enter' message - click the Search button below for results")
    with col2:
        num_results = st.selectbox("Max results:", [3, 5, 10], index=1)
    
    # Search button and options
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search_btn = st.button("🔍 Search", type="primary")
    with col2:
        st.write("")  # Empty space or other content
    with col3:
        ai = get_ai_assistant()
        use_ai = st.checkbox("🤖 AI Analysis", value=ai.is_ready(), disabled=not ai.is_ready())
    if search_btn and query:
        # Perform search
        with st.spinner("Searching..."):
            results = engine.search(query, n_results=num_results)
        
        if results:
            display_search_results(results, query)
            
            # AI Analysis
            if use_ai and results:
                st.divider()
                render_ai_analysis(query, results)
        else:
            st.warning("No results found. Try different search terms or check if documents are loaded.")


def render_ai_analysis(query: str, results: List[Dict]):
    """Render AI analysis of search results"""
    st.subheader("🤖 AI Analysis")
    
    ai = get_ai_assistant()
    
    if not ai.is_ready():
        st.error("AI assistant not available. Check API key configuration.")
        return
    
    # Analysis options
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("AI analysis of search results using Harold Cohen expertise:")
    with col2:
        max_docs = st.selectbox("Analyze top:", [3, 5], index=0, key="ai_max_docs")
    
    with st.spinner("🤖 Analyzing results with AI..."):
        analysis = ai.analyze_search_results(query, results, max_results=max_docs)
    
    if analysis["status"] == "success":
        # Display AI response
        st.write(analysis["response"])
        
        # Show analysis info
        with st.expander("📊 Analysis Details"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Documents Analyzed", analysis["documents_analyzed"])
            with col2:
                st.metric("Cost (this query)", f"${analysis['cost_estimate']:.4f}")
            with col3:
                st.metric("Total Session Cost", f"${analysis['total_cost']:.3f}")
        
        # Generate follow-up suggestions
        if st.button("💡 Suggest Follow-up Questions"):
            with st.spinner("Generating suggestions..."):
                try:
                    suggestions = ai.suggest_follow_up_queries(query, analysis["response"])
                    
                    if suggestions:
                        st.write("**Suggested follow-up research questions:**")
                        for i, suggestion in enumerate(suggestions, 1):
                            if st.button(f"🔍 {suggestion}", key=f"suggestion_{i}"):
                                # Trigger new search with suggested query
                                st.session_state.suggested_query = suggestion
                                st.rerun()
                    else:
                        st.warning("No suggestions generated. Try a different query.")
                        
                except Exception as e:
                    st.error(f"Error generating suggestions: {e}")
    else:
        st.error(f"AI analysis failed: {analysis.get('message', 'Unknown error')}")


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
    
    # Add a note about the refresh behavior
    st.info("💡 **Note:** When you select a file, the page will refresh. Just click back to this Upload tab to continue.")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with documents/emails",
        type=['csv'],
        help="CSV should contain a column with text content (emails, documents, etc.)",
        key="main_file_uploader"
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
    def main():
    """Main application function"""

    # Check password first
    if not check_password():
        return

    # Render header
    render_header()
    
    # Navigation - simple tabs without complex state management
    tab1, tab2, tab3 = st.tabs(["🔍 Search", "📤 Upload", "📊 Overview"])
    
    with tab1:
        render_search_interface()
    
    with tab2:
        render_upload_interface()
    
    with tab3:
        render_collection_overview()


if __name__ == "__main__":
    main()
