#!/usr/bin/env python3
"""
Harold Cohen Catalogue Raisonné Streamlit App
Multi-modal research and archival system
"""

import streamlit as st

# Page configuration MUST be first
st.set_page_config(
    page_title="Harold Cohen Catalogue Raisonné",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import everything else
import pandas as pd
import os
from pathlib import Path
from datetime import datetime
import json
from typing import List, Dict, Any, Optional
import base64
from PIL import Image
import io

# Import your modules (adjust paths as needed)
try:
    from semantic_search import SemanticSearchEngine
    from rag import ResponseGenerator
except ImportError:
    st.error("Could not import semantic_search or rag modules. Please ensure they're in the same directory.")
    st.stop()# Temporarily disabled for deployment

st.session_state.demo_mode = True



if 'search_engine' not in st.session_state:
    st.session_state.search_engine = SemanticSearchEngine()
if 'rag_generator' not in st.session_state:
    st.session_state.rag_generator = ResponseGenerator()


if 'task_list' not in st.session_state:
    st.session_state.task_list = []
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'total_cost' not in st.session_state:
    st.session_state.total_cost = 0.0

def load_task_list():
    """Load task list from file"""
    task_file = Path("hc_tasks.json")
    if task_file.exists():
        with open(task_file, 'r') as f:
            st.session_state.task_list = json.load(f)

def save_task_list():
    """Save task list to file"""
    task_file = Path("hc_tasks.json")
    with open(task_file, 'w') as f:
        json.dump(st.session_state.task_list, f, indent=2)

def add_task(title: str, description: str, category: str = "General"):
    """Add a new task"""
    task = {
        'id': len(st.session_state.task_list) + 1,
        'title': title,
        'description': description,
        'category': category,
        'created': datetime.now().isoformat(),
        'completed': False,
        'related_queries': []
    }
    st.session_state.task_list.append(task)
    save_task_list()

def estimate_cost(text: str, model: str = "haiku") -> float:
    """Rough cost estimation"""
    tokens = len(text) / 4  # Rough approximation
    if model == "haiku":
        return tokens * 0.00025 / 1000  # $0.25 per 1M tokens
    else:  # sonnet
        return tokens * 0.003 / 1000   # $3 per 1M tokens



def check_password():
    """Returns True if the user has entered the correct password."""
    
    # Return True if password is already correct
    if st.session_state.get("password_correct", False):
        return True

    # Show password input
    st.markdown("### 🔒 Harold Cohen Catalogue Raisonné")
    st.markdown("*Please enter the access password*")
    
    # Simple password input
    user_password = st.text_input(
        "Password", 
        type="password",
        placeholder="Enter password here"
    )
    
    # Check password when user types something
    if user_password:
        expected_password = st.secrets.get("APP_PASSWORD", "harold_cohen_2025")
        if user_password == expected_password:
            st.session_state["password_correct"] = True
            st.rerun()  # Refresh to show main app
        else:
            st.error("😕 Password incorrect")
            #st.write(f"Debug: Try '{expected_password}'")  # Temporary debug
    
    st.markdown("---")
    st.markdown("*This system contains private correspondence, inventory data, and confidential research materials.*")
    
    return False

def main():
    # Check password before showing anything
    if not check_password():
        return
    
    st.title("🎨 Harold Cohen Catalogue Raisonné")
    st.markdown("*Comprehensive archival and research system for Harold Cohen's figurative period*")
    
    # Load tasks on startup
    if not st.session_state.task_list:
        load_task_list()
    
    # Sidebar for navigation and settings
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox("Select Page", [
            "🔍 Search & Query",
            "📤 Add New Materials", 
            "📋 Task Management",
            "📊 Collection Overview",
            "⚙️ Settings"
        ])
        
        st.divider()
        
        # Quick stats
        # stats = st.session_state.search_engine.get_collection_stats()
        # st.metric("Documents in Collection", stats.get('total_documents', 0))

        # Quick stats (demo mode when search_engine is None)
        if st.session_state.search_engine:
            stats = st.session_state.search_engine.get_collection_stats()
            st.metric("Documents in Collection", stats.get('total_documents', 0))
        else:
            st.metric("Documents in Collection", "2,847 (demo)")
        st.metric("Session Cost", f"${st.session_state.total_cost:.3f}")
        
        # Quick task summary
        incomplete_tasks = [t for t in st.session_state.task_list if not t['completed']]
        st.metric("Open Tasks", len(incomplete_tasks))
    
    # Main content based on selected page
    if page == "🔍 Search & Query":
        search_and_query_page()
    elif page == "📤 Add New Materials":
        add_materials_page()
    elif page == "📋 Task Management":
        task_management_page()
    elif page == "📊 Collection Overview":
        collection_overview_page()
    elif page == "⚙️ Settings":
        settings_page()

def search_and_query_page():
    st.header("Search & Query")
    
    # Query input
    query = st.text_area(
        "Ask a question about Harold Cohen's work:",
        placeholder="e.g., Do you remember the painting at the National Theater? What happened with it?",
        height=100
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_button = st.button("🔍 Search", type="primary")
    
    with col2:
        num_results = st.selectbox("Results", [2, 3, 5, 10], index=0)
    
    with col3:
        use_ai = st.checkbox("Generate AI Response", value=True)
    
    if search_button and query:
        with st.spinner("Searching..."):
            # Perform semantic search
            raw_results = st.session_state.search_engine.collection.query(
                query_texts=[query],
                n_results=num_results
            )
            
            formatted_results = st.session_state.rag_generator.format_search_results(raw_results)
            
        # Display search results
        st.subheader(f"📄 Found {len(formatted_results)} relevant passages")
        
        for i, result in enumerate(formatted_results, 1):
            with st.expander(f"Result {i}: {result['metadata'].get('source_file', 'Unknown')[:50]}..."):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write("**Content:**")
                    st.write(result['content'][:500] + "..." if len(result['content']) > 500 else result['content'])
                
                with col2:
                    st.write("**Metadata:**")
                    for key, value in result['metadata'].items():
                        if key in ['date', 'sender', 'subject', 'source_type']:
                            st.write(f"**{key}:** {value}")
        
        # Generate AI response if requested
        if use_ai and formatted_results:
            st.divider()
            st.subheader("🤖 AI Analysis")
            
            with st.spinner("Generating response..."):
                # Cost estimation
                context_text = "\n".join([r['content'] for r in formatted_results[:3]])
                estimated_cost = estimate_cost(context_text + query)
                
                st.info(f"Estimated cost: ${estimated_cost:.4f}")
                
                response = st.session_state.rag_generator.generate_response(
                    query=query,
                    context_chunks=formatted_results[:3],
                    max_chunks=3,
                    max_chars_per_chunk=800,
                    use_cheaper_model=True
                )

                st.write(response)
                
                # Update cost tracking
                st.session_state.total_cost += estimated_cost
                
                # Save to history
                st.session_state.query_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'query': query,
                    'results_count': len(formatted_results),
                    'cost': estimated_cost,
                    'response': response[:200] + "..." if len(response) > 200 else response
                })
        
        # Quick task creation
        st.divider()
        st.subheader("📝 Create Task from This Query")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            task_title = st.text_input("Task title", value=query[:50] + "..." if len(query) > 50 else query)
        
        with col2:
            if st.button("Add Task"):
                add_task(
                    title=task_title,
                    description=f"Follow up on query: {query}",
                    category="Research"
                )
                st.success("Task added!")
                st.rerun()

def add_materials_page():
    st.header("Add New Materials")
    
    tab1, tab2, tab3 = st.tabs(["📧 Email Files", "🖼️ Images", "📄 Documents"])
    
    with tab1:
        st.subheader("Upload Email CSV Files")
        uploaded_file = st.file_uploader(
            "Choose CSV file containing emails",
            type=['csv'],
            help="CSV should contain columns like 'message_text', 'sender', 'date', etc."
        )
        
        if uploaded_file:
            # Save uploaded file
            file_path = Path(f"./uploads/{uploaded_file.name}")
            file_path.parent.mkdir(exist_ok=True)
            
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"File saved as {file_path}")
            
            # Preview the file
            df = pd.read_csv(file_path)
            st.write("**Preview:**")
            st.dataframe(df.head())
            
            # Ingestion options
            text_column = st.selectbox("Text column", df.columns.tolist())
            force_reingest = st.checkbox("Force re-ingestion if already exists")
            
            if st.button("Ingest into Collection"):
                with st.spinner("Ingesting emails..."):
                    try:
                        stats = st.session_state.search_engine.ingest_csv_emails(
                            str(file_path),
                            text_column=text_column,
                            force_reingest=force_reingest
                        )
                        
                        if stats['status'] == 'ingested':
                            st.success(f"✅ Ingested {stats['processed_emails']} emails ({stats['total_chunks']} chunks)")
                            
                            # Create a task for this new material
                            add_task(
                                title=f"Review new emails: {uploaded_file.name}",
                                description=f"New email corpus ingested with {stats['processed_emails']} emails",
                                category="New Material"
                            )
                        else:
                            st.info(f"ℹ️ {stats['message']}")
                            
                    except Exception as e:
                        st.error(f"Error during ingestion: {e}")
    
    with tab2:
        st.subheader("Upload Images")
        st.info("🚧 Image handling coming soon! Will support artwork photos, documents, etc.")
        
        uploaded_images = st.file_uploader(
            "Choose image files",
            type=['jpg', 'jpeg', 'png', 'tiff'],
            accept_multiple_files=True
        )
        
        if uploaded_images:
            for img_file in uploaded_images:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    image = Image.open(img_file)
                    st.image(image, caption=img_file.name, width=200)
                
                with col2:
                    st.text_input(f"Title for {img_file.name}", key=f"title_{img_file.name}")
                    st.text_area(f"Description", key=f"desc_{img_file.name}")
                    st.selectbox(f"Category", ["Artwork", "Document", "Reference"], key=f"cat_{img_file.name}")
    
    with tab3:
        st.subheader("Upload Text Documents")
        st.info("🚧 Document handling coming soon! Will support PDFs, Word docs, etc.")

def task_management_page():
    st.header("Task Management")
    
    # Add new task
    with st.expander("➕ Add New Task"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            new_title = st.text_input("Task title")
            new_description = st.text_area("Description")
        
        with col2:
            new_category = st.selectbox("Category", [
                "Research", "Acquisition", "Documentation", 
                "Exhibition", "Conservation", "General"
            ])
            
            if st.button("Add Task") and new_title:
                add_task(new_title, new_description, new_category)
                st.success("Task added!")
                st.rerun()
    
    # Display tasks
    if st.session_state.task_list:
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_completed = st.checkbox("Show completed tasks")
        
        with col2:
            filter_category = st.selectbox("Filter by category", 
                ["All"] + list(set(t['category'] for t in st.session_state.task_list)))
        
        # Filter tasks
        filtered_tasks = st.session_state.task_list
        
        if not show_completed:
            filtered_tasks = [t for t in filtered_tasks if not t['completed']]
        
        if filter_category != "All":
            filtered_tasks = [t for t in filtered_tasks if t['category'] == filter_category]
        
        # Display tasks
        for task in filtered_tasks:
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    if task['completed']:
                        st.write(f"~~{task['title']}~~")
                    else:
                        st.write(f"**{task['title']}**")
                    st.write(task['description'])
                
                with col2:
                    st.write(f"**{task['category']}**")
                    st.write(task['created'][:10])
                
                with col3:
                    if st.button("✅" if not task['completed'] else "↺", key=f"toggle_{task['id']}"):
                        task['completed'] = not task['completed']
                        save_task_list()
                        st.rerun()
                
                with col4:
                    if st.button("🗑️", key=f"delete_{task['id']}"):
                        st.session_state.task_list.remove(task)
                        save_task_list()
                        st.rerun()
                
                st.divider()
    else:
        st.info("No tasks yet. Add some tasks to track your research!")

def collection_overview_page():
    st.header("Collection Overview")
    
    # Get collection statistics
    stats = st.session_state.search_engine.get_collection_stats()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Documents", stats.get('total_documents', 0))
    
    with col2:
        st.metric("Source Types", len(stats.get('source_types', [])))
    
    with col3:
        st.metric("Total Session Cost", f"${st.session_state.total_cost:.3f}")
    
    # Ingested corpora
    st.subheader("📚 Ingested Corpora")
    corpora = stats.get('ingested_corpora', [])
    
    if corpora:
        df = pd.DataFrame(corpora)
        st.dataframe(df)
    else:
        st.info("No corpora ingested yet.")
    
    # Query history
    st.subheader("🔍 Recent Queries")
    if st.session_state.query_history:
        history_df = pd.DataFrame(st.session_state.query_history[-10:])  # Last 10 queries
        st.dataframe(history_df)
    else:
        st.info("No queries yet.")

def settings_page():
    st.header("Settings")
    
    # API Settings
    st.subheader("🔧 API Configuration")
    
    current_key = os.getenv('ANTHROPIC_API_KEY', '')
    masked_key = f"sk-ant-...{current_key[-8:]}" if current_key else "Not set"
    st.write(f"**Current API Key:** {masked_key}")
    
    # Model preferences
    st.subheader("🤖 Model Preferences")
    default_model = st.selectbox("Default model", ["Claude Haiku (Cheap)", "Claude Sonnet (Better)"])
    max_chunks = st.slider("Max context chunks", 1, 10, 3)
    max_chars = st.slider("Max characters per chunk", 500, 2000, 800)
    
    # Collection management
    st.subheader("💾 Collection Management")
    
    if st.button("🗑️ Clear Query History"):
        st.session_state.query_history = []
        st.success("Query history cleared!")
    
    if st.button("💰 Reset Cost Counter"):
        st.session_state.total_cost = 0.0
        st.success("Cost counter reset!")
    
    # Export/Import
    st.subheader("📤 Export/Import")
    
    if st.button("Export Tasks"):
        task_json = json.dumps(st.session_state.task_list, indent=2)
        st.download_button(
            "Download tasks.json",
            task_json,
            "hc_tasks.json",
            "application/json"
        )

if __name__ == "__main__":
    main()
