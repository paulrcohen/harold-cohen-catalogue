#!/usr/bin/env python3
"""
Harold Cohen Catalogue Raisonné Streamlit App
Clean version with improved error handling and debugging
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
from PIL import Image

# Import your modules
try:
    from semantic_search import SemanticSearchEngine
    from rag import ResponseGenerator
except ImportError:
    st.error("Could not import semantic_search or rag modules. Please ensure they're in the same directory.")
    st.stop()

# Initialize session state variables
if 'task_list' not in st.session_state:
    st.session_state.task_list = []
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'total_cost' not in st.session_state:
    st.session_state.total_cost = 0.0

# Initialize search engine with error handling
if 'search_engine' not in st.session_state:
    try:
        with st.spinner("Initializing search engine..."):
            st.session_state.search_engine = SemanticSearchEngine()
            st.session_state.search_available = True
    except Exception as e:
        st.session_state.search_engine = None
        st.session_state.search_available = False
        print(f"Search engine initialization failed: {e}")

# Initialize RAG generator with error handling
if 'rag_generator' not in st.session_state:
    try:
        api_key = st.secrets.get('ANTHROPIC_API_KEY')
        st.session_state.rag_generator = ResponseGenerator(anthropic_api_key=api_key)
        st.session_state.rag_available = True
        st.session_state.rag_error = None
    except Exception as e:
        st.session_state.rag_generator = ResponseGenerator()
        st.session_state.rag_available = False
        st.session_state.rag_error = str(e)
        print(f"RAG generator initialization failed: {e}")


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
    
    st.markdown("---")
    st.markdown("*This system contains private correspondence, inventory data, and confidential research materials.*")
    
    return False

def debug_collection_page():
    """Debug page to check collection status and add test data"""
    st.header("🔍 Collection Debug & Testing")
    
    if not st.session_state.get('search_available', False):
        st.error("Search engine not available")
        return
    
    # Show persistence status
    st.subheader("💾 Persistence Status")
    try:
        status = st.session_state.search_engine.get_persistence_status()
        
        col1, col2 = st.columns(2)
        with col1:
            if status["using_persistent"]:
                st.success("✅ Using Persistent Storage")
            else:
                st.error("❌ Using In-Memory Storage")
        
        with col2:
            st.info(f"Documents: {status.get('document_count', 0)}")
        
        # Show detailed status
        with st.expander("Detailed Persistence Info"):
            for key, value in status.items():
                st.write(f"**{key}:** {value}")
                
        # Warning for in-memory storage
        if not status["using_persistent"]:
            st.warning("""
            ⚠️ **Data Persistence Issue Detected**
            
            ChromaDB is running in memory-only mode. This means:
            - All data will be lost when the app restarts
            - You'll need to re-ingest documents each session
            - Test data has been automatically added for this session
            """)
            
    except Exception as e:
        st.error(f"Could not get persistence status: {e}")
    
    st.divider()
    
    # Add test data manually
    st.subheader("➕ Add More Test Data")
    if st.button("Add Additional Test Documents", key="add_more_test_data"):
        try:
            additional_docs = [
                "Cohen's AARON program evolved significantly during the 1980s figurative period. The system learned to create more complex human forms and spatial relationships.",
                "We received confirmation that all artwork arrived at the museum in good condition. The insurance documentation has been filed appropriately.",
                "The exhibition opening was a great success. Many visitors were fascinated by the computer-generated artwork and Cohen's innovative approach."
            ]
            
            additional_metadata = [
                {"source_type": "research", "source_file": "aaron_evolution.txt"},
                {"source_type": "email", "source_file": "shipping_confirmation.txt", "sender": "insurance@artcare.com"},
                {"source_type": "event", "source_file": "opening_notes.txt", "event": "exhibition_opening"}
            ]
            
            additional_ids = [f"additional_doc_{i}" for i in range(len(additional_docs))]
            
            st.session_state.search_engine.collection.add(
                documents=additional_docs,
                metadatas=additional_metadata,
                ids=additional_ids
            )
            
            st.success(f"Added {len(additional_docs)} additional test documents")
            st.rerun()
            
        except Exception as e:
            st.error(f"Failed to add additional test data: {e}")
    
    st.divider()
    
    # Collection statistics
    st.subheader("📊 Collection Statistics")
    try:
        stats = st.session_state.search_engine.get_collection_stats()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Documents", stats.get('total_documents', 0))
        with col2:
            st.metric("Source Types", len(stats.get('source_types', [])))
        with col3:
            status_color = "✅" if stats.get('total_documents', 0) > 0 else "❌"
            st.metric("Status", f"{status_color} {stats.get('status', 'unknown')}")
        
        # Show source types
        if stats.get('source_types'):
            st.write("**Source Types:**", ", ".join(stats['source_types']))
        
    except Exception as e:
        st.error(f"Error getting collection stats: {e}")
    
    st.divider()
    
    # Test search
    st.subheader("🧪 Test Search")
    test_query = st.text_input(
        "Try a search query:", 
        value="Brooklyn museum",
        help="Try: 'Brooklyn museum', 'shipping delays', 'National Theatre', or 'Harold Cohen'",
        key="debug_test_query"
    )
    
    if st.button("Test Search", key="debug_test_search") and test_query:
        with st.spinner("Searching..."):
            try:
                results = st.session_state.search_engine.search(test_query, n_results=3)
                
                if results:
                    st.success(f"✅ Found {len(results)} results!")
                    
                    for i, result in enumerate(results, 1):
                        with st.expander(f"Result {i}"):
                            st.write("**Content:**")
                            st.write(result['content'][:300] + "..." if len(result['content']) > 300 else result['content'])
                            st.write("**Metadata:**")
                            for key, value in result['metadata'].items():
                                st.write(f"- **{key}:** {value}")
                            if result.get('distance') is not None:
                                st.write(f"**Distance:** {result['distance']:.3f}")
                else:
                    st.warning("❌ No results found")
                    
            except Exception as e:
                st.error(f"Search failed: {e}")

def search_and_query_page():
    """Main search and query page"""
    st.header("Search & Query")
    
    # Check if search is available
    if not st.session_state.get('search_available', False):
        st.error("❌ Search functionality is currently unavailable.")
        st.info("Please check the Debug Collection page for more information.")
        return
    
    # Show data status
    try:
        stats = st.session_state.search_engine.get_collection_stats()
        if stats.get('total_documents', 0) == 0:
            st.warning("⚠️ No documents in collection. Please add data via 'Add New Materials' or use 'Debug Collection' to add test data.")
            return
    except:
        pass
    
    # Query input
    query = st.text_area(
        "Ask a question about Harold Cohen's work:",
        placeholder="e.g., Do you remember the painting at the National Theater? What happened with it?",
        height=100,
        key="main_query_input"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_button = st.button("🔍 Search", type="primary", key="main_search_button")
    
    with col2:
        num_results = st.selectbox("Results", [2, 3, 5, 10], index=1, key="num_results_select")
    
    with col3:
        if st.session_state.get('rag_available', False):
            use_ai = st.checkbox("Generate AI Response", value=True, key="use_ai_checkbox")
        else:
            use_ai = False
            st.info("AI unavailable")
    
    if search_button and query:
        with st.spinner("Searching..."):
            try:
                # Use the search method from SemanticSearchEngine
                results = st.session_state.search_engine.search(query, n_results=num_results)
                
            except Exception as e:
                st.error(f"Search failed: {str(e)}")
                return
        
        # Display search results
        if results:
            st.subheader(f"📄 Found {len(results)} relevant passages")
            
            for i, result in enumerate(results, 1):
                with st.expander(f"Result {i}: {result['metadata'].get('source_file', 'Unknown')[:50]}...", key=f"result_expander_{i}"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write("**Content:**")
                        content = result['content']
                        st.write(content[:500] + "..." if len(content) > 500 else content)
                    
                    with col2:
                        st.write("**Metadata:**")
                        for key, value in result['metadata'].items():
                            if key in ['date', 'sender', 'subject', 'source_type', 'title']:
                                st.write(f"**{key}:** {value}")
                        
                        if result.get('distance') is not None:
                            st.write(f"**Relevance:** {1-result['distance']:.2f}")
        else:
            st.warning("No results found. Try different search terms.")
            return
        
        # Generate AI response if requested
        if use_ai and results and st.session_state.get('rag_available', False):
            st.divider()
            st.subheader("🤖 AI Analysis")
            
            with st.spinner("Generating response..."):
                try:
                    # Estimate cost
                    context_text = "\n".join([r['content'] for r in results[:3]])
                    estimated_cost = estimate_cost(context_text + query)
                    
                    st.info(f"Estimated cost: ${estimated_cost:.4f}")
                    
                    # Generate response using the formatted results
                    response = st.session_state.rag_generator.generate_response(
                        query=query,
                        context_chunks=results[:3],
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
                        'results_count': len(results),
                        'cost': estimated_cost,
                        'response': response[:200] + "..." if len(response) > 200 else response
                    })
                    
                except Exception as e:
                    st.error(f"AI response generation failed: {str(e)}")
        
        # Quick task creation
        st.divider()
        st.subheader("📝 Create Task from This Query")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            task_title = st.text_input(
                "Task title", 
                value=query[:50] + "..." if len(query) > 50 else query,
                key="task_title_input"
            )
        
        with col2:
            if st.button("Add Task", key="add_task_button"):
                add_task(
                    title=task_title,
                    description=f"Follow up on query: {query}",
                    category="Research"
                )
                st.success("Task added!")
                st.rerun()

def add_materials_page():
    """Page for adding new materials"""
    st.header("Add New Materials")
    
    if not st.session_state.get('search_available', False):
        st.error("Search engine not available - cannot ingest materials")
        return
    
    tab1, tab2 = st.tabs(["📧 Email Files", "🖼️ Images"])
    
    with tab1:
        st.subheader("Upload Email CSV Files")
        uploaded_file = st.file_uploader(
            "Choose CSV file containing emails",
            type=['csv'],
            help="CSV should contain columns like 'message_text', 'sender', 'date', etc.",
            key="csv_uploader"
        )
        
        if uploaded_file:
            # Save uploaded file
            file_path = Path(f"./uploads/{uploaded_file.name}")
            file_path.parent.mkdir(exist_ok=True)
            
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"File saved as {file_path}")
            
            # Preview the file
            try:
                df = pd.read_csv(file_path)
                st.write("**Preview:**")
                st.dataframe(df.head())
                
                # Ingestion options
                text_column = st.selectbox("Text column", df.columns.tolist(), key="text_column_select")
                force_reingest = st.checkbox("Force re-ingestion if already exists", key="force_reingest_checkbox")
                
                if st.button("Ingest into Collection", key="ingest_button"):
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
                                st.rerun()
                            else:
                                st.info(f"ℹ️ {stats['message']}")
                                
                        except Exception as e:
                            st.error(f"Error during ingestion: {e}")
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
    
    with tab2:
        st.subheader("Upload Images")
        st.info("🚧 Image handling coming soon! Will support artwork photos, documents, etc.")
        
        uploaded_images = st.file_uploader(
            "Choose image files",
            type=['jpg', 'jpeg', 'png', 'tiff'],
            accept_multiple_files=True,
            key="image_uploader"
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

def task_management_page():
    """Task management page"""
    st.header("Task Management")
    
    # Add new task
    with st.expander("➕ Add New Task"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            new_title = st.text_input("Task title", key="new_task_title")
            new_description = st.text_area("Description", key="new_task_description")
        
        with col2:
            new_category = st.selectbox("Category", [
                "Research", "Acquisition", "Documentation", 
                "Exhibition", "Conservation", "General"
            ], key="new_task_category")
            
            if st.button("Add Task", key="add_new_task_button") and new_title:
                add_task(new_title, new_description, new_category)
                st.success("Task added!")
                st.rerun()
    
    # Display tasks
    if st.session_state.task_list:
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_completed = st.checkbox("Show completed tasks", key="show_completed_filter")
        
        with col2:
            filter_category = st.selectbox("Filter by category", 
                ["All"] + list(set(t['category'] for t in st.session_state.task_list)),
                key="category_filter")
        
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
    """Collection overview page"""
    st.header("Collection Overview")
    
    if not st.session_state.get('search_available', False):
        st.error("Search engine not available")
        return
    
    # Get collection statistics
    try:
        stats = st.session_state.search_engine.get_collection_stats()
        persistence_status = st.session_state.search_engine.get_persistence_status()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Documents", stats.get('total_documents', 0))
        
        with col2:
            st.metric("Source Types", len(stats.get('source_types', [])))
        
        with col3:
            st.metric("Total Session Cost", f"${st.session_state.total_cost:.3f}")
        
        with col4:
            if persistence_status.get('using_persistent', False):
                st.metric("Storage", "💾 Persistent")
            else:
                st.metric("Storage", "⚠️ Temporary")
        
        # Show source types
        if stats.get('source_types'):
            st.subheader("📚 Source Types")
            source_types = stats['source_types']
            cols = st.columns(len(source_types))
            for i, source_type in enumerate(source_types):
                with cols[i]:
                    st.info(f"**{source_type}**")
        
        # Persistence warning
        if not persistence_status.get('using_persistent', False):
            st.warning("""
            ⚠️ **Temporary Storage Active**
            
            Your data is stored in memory only and will be lost when the app restarts.
            Consider uploading your data files for each session, or check the Debug Collection page for more details.
            """)
        
        # Query history
        st.subheader("🔍 Recent Queries")
        if st.session_state.query_history:
            history_df = pd.DataFrame(st.session_state.query_history[-10:])  # Last 10 queries
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp']).dt.strftime('%H:%M:%S')
            st.dataframe(history_df[['timestamp', 'query', 'results_count', 'cost']])
        else:
            st.info("No queries yet.")
            
    except Exception as e:
        st.error(f"Error getting collection overview: {e}")

def settings_page():
    """Settings page"""
    st.header("Settings")
    
    # System Status
    st.subheader("🔧 System Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Search Engine:**")
        if st.session_state.get('search_available', False):
            st.success("✅ Available")
            try:
                status = st.session_state.search_engine.get_persistence_status()
                if status.get('using_persistent', False):
                    st.info("💾 Using persistent storage")
                else:
                    st.warning("⚠️ Using temporary storage")
            except:
                pass
        else:
            st.error("❌ Unavailable")
    
    with col2:
        st.write("**AI Response Generator:**")
        if st.session_state.get('rag_available', False):
            st.success("✅ Available")
        else:
            st.error("❌ Unavailable")
            if st.session_state.get('rag_error'):
                with st.expander("Error Details"):
                    st.code(st.session_state.rag_error)
    
    # API Settings
    st.subheader("🔑 API Configuration")
    
    current_key = os.getenv('ANTHROPIC_API_KEY', '')
    if current_key:
        masked_key = f"sk-ant-...{current_key[-8:]}"
        st.success(f"✅ API Key: {masked_key}")
    else:
        st.warning("⚠️ No API key configured")
    
    # Model preferences
    st.subheader("🤖 Model Preferences")
    default_model = st.selectbox("Default model", ["Claude Haiku (Cheap)", "Claude Sonnet (Better)"], key="model_preference")
    max_chunks = st.slider("Max context chunks", 1, 10, 3, key="max_chunks_setting")
    max_chars = st.slider("Max characters per chunk", 500, 2000, 800, key="max_chars_setting")
    
    # Data management
    st.subheader("💾 Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Query History", key="clear_query_history"):
            st.session_state.query_history = []
            st.success("Query history cleared!")
            st.rerun()
    
    with col2:
        if st.button("💰 Reset Cost Counter", key="reset_cost_counter"):
            st.session_state.total_cost = 0.0
            st.success("Cost counter reset!")
            st.rerun()
    
    # Export/Import
    st.subheader("📤 Export/Import")
    
    if st.button("Export Tasks", key="export_tasks_button"):
        task_json = json.dumps(st.session_state.task_list, indent=2)
        st.download_button(
            "Download tasks.json",
            task_json,
            "hc_tasks.json",
            "application/json",
            key="download_tasks_button"
        )

def main():
    """Main application function"""
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
        # System status indicators
        st.header("System Status")
        
        # Search engine status
        if st.session_state.get('search_available', False):
            st.success("🔍 Search: Available")
        else:
            st.error("🔍 Search: Unavailable")
        
        # RAG status  
        if st.session_state.get('rag_available', False):
            st.success("🤖 AI: Available")
        else:
            st.warning("🤖 AI: Unavailable")
        
        st.divider()
        
        # Navigation
        st.header("Navigation")
        page = st.selectbox("Select Page", [
            "🔍 Search & Query",
            "📤 Add New Materials", 
            "📋 Task Management",
            "📊 Collection Overview",
            "🐛 Debug Collection",
            "⚙️ Settings"
        ], key="main_page_selector")
        
        st.divider()
        
        # Quick stats
        try:
            if st.session_state.get('search_available', False):
                stats = st.session_state.search_engine.get_collection_stats()
                st.metric("Documents in Collection", stats.get('total_documents', 0))
            else:
                st.metric("Documents in Collection", "N/A")
        except Exception:
            st.metric("Documents in Collection", "Error")
            
        st.metric("Session Cost", f"${st.session_state.total_cost:.3f}")
        
        # Quick task summary
        incomplete_tasks = [t for t in st.session_state.task_list if not t['completed']]
        st.metric("Open Tasks", len(incomplete_tasks))
    
    # Main content based on selected page
    try:
        if page == "🔍 Search & Query":
            search_and_query_page()
        elif page == "📤 Add New Materials":
            add_materials_page()
        elif page == "📋 Task Management":
            task_management_page()
        elif page == "📊 Collection Overview":
            collection_overview_page()
        elif page == "🐛 Debug Collection":
            debug_collection_page()
        elif page == "⚙️ Settings":
            settings_page()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please try refreshing the page or contact support if the problem persists.")
        
        # Show error details in expander for debugging
        with st.expander("Error Details (for debugging)"):
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
