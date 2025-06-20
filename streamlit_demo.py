#!/usr/bin/env python3
"""
Simplified Harold Cohen Catalogue Raisonné Streamlit App
Focus on core functionality without excessive diagnostics
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

# Page config must be first
st.set_page_config(
    page_title="Harold Cohen Catalogue Raisonné",
    page_icon="🎨",
    layout="wide"
)

# Import modules with error handling
try:
    from semantic_search import SemanticSearchEngine
    search_module_available = True
except ImportError as e:
    st.error(f"Could not import semantic_search module: {e}")
    search_module_available = False

try:
    from rag import ResponseGenerator
    rag_module_available = True
except ImportError:
    rag_module_available = False

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.query_history = []
    st.session_state.total_cost = 0.0

# Initialize search engine
if 'search_engine' not in st.session_state:
    if search_module_available:
        try:
            with st.spinner("Initializing search engine..."):
                st.session_state.search_engine = SemanticSearchEngine()
                st.session_state.search_ready = True
        except Exception as e:
            st.error(f"Search engine initialization failed: {e}")
            st.session_state.search_engine = None
            st.session_state.search_ready = False
    else:
        st.session_state.search_engine = None
        st.session_state.search_ready = False

# Initialize RAG
if 'rag_generator' not in st.session_state and rag_module_available:
    try:
        api_key = st.secrets.get('ANTHROPIC_API_KEY')
        if api_key:
            st.session_state.rag_generator = ResponseGenerator(anthropic_api_key=api_key)
            st.session_state.rag_ready = True
        else:
            st.session_state.rag_ready = False
    except Exception as e:
        st.session_state.rag_ready = False

def check_password():
    """Simple password check"""
    if st.session_state.get("password_correct", False):
        return True
    
    st.markdown("### 🔒 Harold Cohen Catalogue Raisonné")
    password = st.text_input("Password", type="password")
    
    if password:
        expected = st.secrets.get("APP_PASSWORD", "harold_cohen_2025")
        if password == expected:
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.error("Incorrect password")
    
    return False

def main_search_page():
    """Main search interface"""
    st.header("🔍 Search & Query")
    
    # Check if search is ready
    if not st.session_state.get('search_ready', False) or not st.session_state.get('search_engine'):
        st.error("Search engine not available")
        if st.session_state.get('search_engine') is None:
            st.info("The semantic_search module may not be properly installed or there was an initialization error.")
        return
    
    # Show status
    status = st.session_state.search_engine.get_status()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Documents", status.get('document_count', 0))
    with col2:
        if status.get('persistent_storage', False):
            st.success("💾 Persistent Storage")
        else:
            st.warning("⚠️ Memory Only")
    with col3:
        st.metric("Session Cost", f"${st.session_state.total_cost:.3f}")
    
    # Search interface
    query = st.text_area(
        "Search the Harold Cohen collection:",
        placeholder="e.g., Brooklyn museum exhibition, National Theatre painting, shipping delays...",
        height=100
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search_btn = st.button("🔍 Search", type="primary")
    with col2:
        num_results = st.selectbox("Results", [3, 5, 10], index=0)
    with col3:
        use_ai = st.checkbox("AI Response", value=st.session_state.get('rag_ready', False))
    
    if search_btn and query:
        # Perform search
        with st.spinner("Searching..."):
            results = st.session_state.search_engine.search(query, n_results=num_results)
        
        if results:
            st.subheader(f"📄 Found {len(results)} results")
            
            # Display results
            for i, result in enumerate(results, 1):
                with st.expander(f"Result {i} - {result['metadata'].get('source_file', 'Unknown source')}"):
                    st.write("**Content:**")
                    st.write(result['content'])
                    
                    if result['metadata']:
                        st.write("**Metadata:**")
                        for key, value in result['metadata'].items():
                            st.write(f"- **{key}:** {value}")
                    
                    if 'distance' in result:
                        relevance = 1 - result['distance']
                        st.write(f"**Relevance:** {relevance:.2f}")
            
            # AI Response
            if use_ai and st.session_state.get('rag_ready', False):
                st.divider()
                st.subheader("🤖 AI Analysis")
                
                with st.spinner("Generating AI response..."):
                    try:
                        response = st.session_state.rag_generator.generate_response(
                            query=query,
                            context_chunks=results[:3],
                            max_chunks=3,
                            use_cheaper_model=True
                        )
                        st.write(response)
                        
                        # Rough cost estimate
                        estimated_cost = len(query + str(results[:3])) * 0.0001 / 1000
                        st.session_state.total_cost += estimated_cost
                        
                    except Exception as e:
                        st.error(f"AI response failed: {e}")
            
            # Save to history
            st.session_state.query_history.append({
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'results_count': len(results)
            })
            
        else:
            st.warning("No results found. Try different search terms.")

def upload_page():
    """File upload page"""
    st.header("📤 Upload Materials")
    
    if not st.session_state.get('search_ready', False) or not st.session_state.get('search_engine'):
        st.error("Search engine not available")
        return
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with emails/documents",
        type=['csv'],
        help="CSV should have a column with text content (e.g., 'message_text', 'content', 'text')"
    )
    
    if uploaded_file:
        # Save file temporarily
        temp_path = Path(f"./temp_uploads/{uploaded_file.name}")
        temp_path.parent.mkdir(exist_ok=True)
        
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        # Preview file
        try:
            df = pd.read_csv(temp_path)
            st.success(f"File uploaded: {len(df)} rows")
            
            # Show preview
            st.write("**Preview:**")
            st.dataframe(df.head())
            
            # Select text column
            text_columns = [col for col in df.columns if any(keyword in col.lower() 
                          for keyword in ['text', 'message', 'content', 'body'])]
            
            if text_columns:
                default_col = text_columns[0]
            else:
                default_col = df.columns[0] if len(df.columns) > 0 else None
            
            text_column = st.selectbox(
                "Select text column:",
                df.columns.tolist(),
                index=df.columns.tolist().index(default_col) if default_col else 0
            )
            
            # Ingest button
            if st.button("📥 Ingest into Collection", type="primary"):
                with st.spinner("Ingesting documents..."):
                    result = st.session_state.search_engine.ingest_csv_emails(
                        str(temp_path), 
                        text_column=text_column
                    )
                    
                    if result['status'] == 'success':
                        st.success(f"✅ Successfully ingested {result['documents_added']} documents!")
                        st.info(f"Total documents in collection: {result['total_documents']}")
                        st.rerun()
                    else:
                        st.error(f"❌ Ingestion failed: {result['message']}")
        
        except Exception as e:
            st.error(f"Error reading file: {e}")

def history_page():
    """Query history page"""
    st.header("📊 Query History")
    
    if st.session_state.query_history:
        # Convert to DataFrame for better display
        df = pd.DataFrame(st.session_state.query_history)
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        st.dataframe(df, use_container_width=True)
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.query_history = []
            st.rerun()
    else:
        st.info("No search history yet.")

def debug_page():
    """Simple debug page"""
    st.header("🔧 System Status")
    
    # Search engine status
    st.subheader("Search Engine")
    if st.session_state.get('search_ready', False) and st.session_state.get('search_engine'):
        status = st.session_state.search_engine.get_status()
        
        col1, col2 = st.columns(2)
        with col1:
            st.success("✅ Search engine active")
            st.write(f"**Documents:** {status.get('document_count', 0)}")
        with col2:
            if status.get('persistent_storage', False):
                st.success("💾 Using persistent storage")
                st.write(f"**Path:** {status.get('storage_path', 'Unknown')}")
            else:
                st.warning("⚠️ Using memory storage")
                st.write("Data will be lost on restart")
    else:
        st.error("❌ Search engine not available")
        if not search_module_available:
            st.write("- semantic_search module could not be imported")
        elif st.session_state.get('search_engine') is None:
            st.write("- Search engine initialization failed")
    
    # RAG status
    st.subheader("AI Response Generator")
    if st.session_state.get('rag_ready', False):
        st.success("✅ AI responses available")
    else:
        st.warning("⚠️ AI responses not available")
        if not rag_module_available:
            st.write("- rag module could not be imported")
        else:
            st.write("- Check API key configuration")
    
    # Test search
    st.subheader("Test Search")
    if st.session_state.get('search_ready', False) and st.session_state.get('search_engine'):
        test_query = st.text_input("Test query:", value="Harold Cohen")
        if st.button("Test") and test_query:
            results = st.session_state.search_engine.search(test_query, n_results=2)
            st.write(f"Found {len(results)} results")
            for r in results:
                st.write(f"- {r['content'][:100]}...")
    else:
        st.info("Search engine not available for testing")

def main():
    """Main app function"""
    if not check_password():
        return
    
    st.title("🎨 Harold Cohen Catalogue Raisonné")
    st.markdown("*Research system for Harold Cohen's figurative period*")
    
    # Navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Search", 
        "📤 Upload", 
        "📊 History", 
        "🔧 Debug"
    ])
    
    with tab1:
        main_search_page()
    
    with tab2:
        upload_page()
    
    with tab3:
        history_page()
    
    with tab4:
        debug_page()

if __name__ == "__main__":
    main()
