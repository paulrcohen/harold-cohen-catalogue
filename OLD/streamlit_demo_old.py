#!/usr/bin/env python3
"""
Harold Cohen Catalogue Raisonné Streamlit App
Multi-modal research and archival system
"""

import streamlit as st

# Page configuration MUST be first
st.set_page_config(
    page_title="Harold Cohen Catalogue Raisonné",
    page_icon="None",
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
        # Create a basic generator that won't use AI
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
    
    # Show system status in sidebar
    with st.sidebar:
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
            if st.session_state.get('rag_error'):
                with st.expander("AI Error Details"):
                    st.code(st.session_state.rag_error)
    
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
            "🐛 Debug Collection",
            "⚙️ Settings"
        ], key="main_page_selector")
        
        st.divider()
        
        # Quick stats with error handling
        try:
            if st.session_state.get('search_available', False):
                stats = st.session_state.search_engine.get_collection_stats()
                st.metric("Documents in Collection", stats.get('total_documents', 0))
            else:
                st.metric("Documents in Collection", "N/A")
        except Exception as e:
            st.metric("Documents in Collection", "Error")
            
        st.metric("Session Cost", f"${st.session_state.total_cost:.3f}")
        
        # Quick task summary
        incomplete_tasks = [t for t in st.session_state.task_list if not t['completed']]
        st.metric("Open Tasks", len(incomplete_tasks))
    
    # Main content based on selected page
    try:
        if page == "🔍 Search & Query":
            if not st.session_state.get('search_available', False):
                st.error("❌ Search functionality is currently unavailable. Please check the system configuration.")
                st.info("This might be due to missing dependencies or configuration issues.")
            else:
                search_and_query_page()
        elif page == "📤 Add New Materials":
            add_materials_page()
        elif page == "📋 Task Management":
            task_management_page()
        elif page == "📊 Collection Overview":
            collection_overview_page()
        elif page == "⚙️ Settings":
            settings_page()
        elif page == "🐛 Debug Collection":
            debug_collection_page()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please try refreshing the page or contact support if the problem persists.")
        
        # Show error details in expander for debugging
        with st.expander("Error Details (for debugging)"):
            import traceback
            st.code(traceback.format_exc())

def search_and_query_page():
    st.header("Search & Query")
    
    # Query input with unique key
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
        num_results = st.selectbox("Results", [2, 3, 5, 10], index=0, key="num_results_select")
    
    with col3:
        # Check if RAG is available before showing the checkbox
        if st.session_state.get('rag_available', False):
            use_ai = st.checkbox("Generate AI Response", value=True, key="use_ai_checkbox")
        else:
            use_ai = False
            st.info("AI unavailable")
    
    if search_button and query:
        # Check if search engine is available
        if not st.session_state.get('search_available', False):
            st.error("Search engine is not available. Please check the system configuration.")
            return
            
        with st.spinner("Searching..."):
            try:
                # Perform semantic search
                raw_results = st.session_state.search_engine.collection.query(
                    query_texts=[query],
                    n_results=num_results
                )
                
                formatted_results = st.session_state.rag_generator.format_search_results(raw_results)
                
            except Exception as e:
                st.error(f"Search failed: {str(e)}")
                return
        
        # Display search results
        st.subheader(f"📄 Found {len(formatted_results)} relevant passages")
        
        for i, result in enumerate(formatted_results, 1):
            with st.expander(f"Result {i}: {result['metadata'].get('source_file', 'Unknown')[:50]}...", key=f"result_expander_{i}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write("**Content:**")
                    st.write(result['content'][:500] + "..." if len(result['content']) > 500 else result['content'])
                
                with col2:
                    st.write("**Metadata:**")
                    for key, value in result['metadata'].items():
                        if key in ['date', 'sender', 'subject', 'source_type']:
                            st.write(f"**{key}:** {value}")
        
        # Generate AI response if requested and available
        if use_ai and formatted_results and st.session_state.get('rag_available', False):
            st.divider()
            st.subheader("🤖 AI Analysis")
            
            with st.spinner("Generating response..."):
                try:
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

def add_test_data_to_collection():
    """Add some test data to the collection for testing"""
    test_documents = [
        {
            "content": "Harold Cohen was a pioneering computer artist who created the AARON drawing program. He worked extensively on computer-generated art from the 1970s onwards. His figurative period began in the early 1980s when AARON started creating human figures.",
            "metadata": {
                "source_file": "test_bio.txt",
                "source_type": "biography",
                "subject": "Harold Cohen biography",
                "date": "2024-01-01"
            }
        },
        {
            "content": "The Brooklyn Museum exhibition of Harold Cohen's work featured several large-scale prints and drawings. We shipped 15 posters from the museum's collection. The installation went smoothly and the curator was very pleased with the presentation.",
            "metadata": {
                "source_file": "test_email1.txt", 
                "source_type": "email",
                "sender": "curator@brooklyn.museum",
                "subject": "Exhibition shipping - Brooklyn Museum",
                "date": "2024-02-15"
            }
        },
        {
            "content": "There were some delays with customs for the international shipment. The paperwork took longer than expected, but everything arrived safely. We had to provide additional documentation about the artwork's provenance and insurance values.",
            "metadata": {
                "source_file": "test_email2.txt",
                "source_type": "email", 
                "sender": "shipping@artlogistics.com",
                "subject": "Customs delay update",
                "date": "2024-02-20"
            }
        },
        {
            "content": "The National Theatre painting was a significant work from Cohen's figurative period. It measured 6 feet by 4 feet and featured his characteristic drawing style with flowing lines and human figures. The painting was completed in 1987 and shows AARON's evolution toward more complex compositions.",
            "metadata": {
                "source_file": "test_artwork.txt",
                "source_type": "artwork_description",
                "title": "National Theatre painting",
                "date": "1987",
                "dimensions": "6x4 feet"
            }
        },
        {
            "content": "Do you remember the painting at the National Theater? I think there was some issue with the installation or maybe the frame. The curator mentioned something about the lighting not being quite right for that particular piece.",
            "metadata": {
                "source_file": "test_email3.txt",
                "source_type": "email",
                "sender": "gallery@nationaltheatre.org",
                "subject": "National Theatre installation question",
                "date": "2024-03-01"
            }
        }
    ]
    
    try:
        collection = st.session_state.search_engine.collection
        
        # Prepare data for ChromaDB
        documents = [doc["content"] for doc in test_documents]
        metadatas = [doc["metadata"] for doc in test_documents]
        ids = [f"test_doc_{i}" for i in range(len(test_documents))]
        
        # Add to collection
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        return True, f"Added {len(test_documents)} test documents"
        
    except Exception as e:
        return False, f"Error adding test data: {e}"

def check_chromadb_persistence():
    """Check ChromaDB persistence configuration"""
    try:
        # Check if search engine exists
        if not hasattr(st.session_state, 'search_engine'):
            return "No search engine found"
        
        search_engine = st.session_state.search_engine
        
        # Check client type
        if hasattr(search_engine, 'client'):
            client = search_engine.client
            client_type = type(client).__name__
            
            # Check if it's persistent
            if hasattr(client, '_settings'):
                settings = client._settings
                persistence_info = f"Client type: {client_type}, Settings: {settings}"
            else:
                persistence_info = f"Client type: {client_type}"
                
            # Check for data directory
            import os
            data_dirs = []
            for possible_dir in ['./chroma_db', './chromadb', './data', '.']:
                if os.path.exists(possible_dir):
                    contents = os.listdir(possible_dir)
                    if any('chroma' in item.lower() for item in contents):
                        data_dirs.append(f"{possible_dir}: {contents}")
            
            return {
                "client_info": persistence_info,
                "data_directories": data_dirs,
                "working_directory": os.getcwd(),
                "client_class": str(type(client))
            }
        else:
            return "No client found in search engine"
            
    except Exception as e:
        return f"Error checking persistence: {e}"

def debug_collection_page():
    """Debug page to check collection status and add test data"""
    st.header("🔍 Collection Debug & Testing")
    
    if not st.session_state.get('search_available', False):
        st.error("Search engine not available")
        return
    
    # Add test data section
    st.subheader("➕ Add Test Data")
    st.info("Since your collection is empty, you can add some test documents to try out the search functionality.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("Test data includes:")
        st.write("- Harold Cohen biography info")
        st.write("- Brooklyn Museum shipping emails") 
        st.write("- Customs delay correspondence")
        st.write("- National Theatre painting details")
    
    with col2:
        if st.button("Add Test Documents", type="primary"):
            with st.spinner("Adding test documents..."):
                success, message = add_test_data_to_collection()
                if success:
                    st.success(message)
                    st.rerun()  # Refresh to show updated stats
                else:
                    st.error(message)
    
    st.divider()

    # Chromadb persistence check
    st.subheader("🔧 Persistence Check")
    persistence_info = check_chromadb_persistence()
    
    if isinstance(persistence_info, dict):
        st.write("**ChromaDB Configuration:**")
        for key, value in persistence_info.items():
            st.write(f"- **{key}:** {value}")
    else:
        st.write(persistence_info)
    
    # Check file system
    st.write("**File System Check:**")
    import os
    current_files = os.listdir('.')
    st.write(f"Current directory contents: {current_files}")

    
    # Collection stats
    st.subheader("📊 Collection Statistics")
    try:
        stats = st.session_state.search_engine.get_collection_stats()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Documents", stats.get('total_documents', 0))
        with col2:
            st.metric("Source Types", len(stats.get('source_types', [])))
        with col3:
            if stats.get('total_documents', 0) > 0:
                st.metric("Status", "✅ Ready")
            else:
                st.metric("Status", "❌ Empty")
        
        # Show detailed stats
        with st.expander("Detailed Statistics"):
            st.json(stats)
            
    except Exception as e:
        st.error(f"Error getting collection stats: {e}")
    
    st.divider()
    
    # Sample documents
    st.subheader("📄 Sample Documents")
    try:
        if hasattr(st.session_state.search_engine, 'collection'):
            collection = st.session_state.search_engine.collection
            
            # Get some sample documents
            sample_query = collection.get(limit=5)
            
            if sample_query and sample_query.get('documents'):
                st.success(f"Found {len(sample_query['documents'])} documents in collection")
                
                for i, doc in enumerate(sample_query['documents']):
                    with st.expander(f"Document {i+1} (ID: {sample_query.get('ids', [f'unknown_{i}'])[i]})"):
                        st.write("**Content:**")
                        st.write(doc[:300] + "..." if len(doc) > 300 else doc)
                        
                        if 'metadatas' in sample_query and i < len(sample_query['metadatas']):
                            st.write("**Metadata:**")
                            metadata = sample_query['metadatas'][i]
                            for key, value in metadata.items():
                                st.write(f"- **{key}:** {value}")
            else:
                st.warning("No documents found in collection")
                
    except Exception as e:
        st.error(f"Error getting sample documents: {e}")
    
    st.divider()
    
    # Test search
    st.subheader("🧪 Test Search")
    test_query = st.text_input(
        "Try a search query:", 
        value="Brooklyn museum posters",
        help="Try: 'Brooklyn museum', 'shipping delays', 'National Theatre', or 'Harold Cohen'",
        key="debug_test_query"
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Test Search", key="debug_test_search") and test_query:
            with st.spinner("Searching..."):
                try:
                    raw_results = st.session_state.search_engine.collection.query(
                        query_texts=[test_query],
                        n_results=3
                    )
                    
                    st.write("**Search Results:**")
                    if raw_results.get('documents') and raw_results['documents'][0]:
                        st.success(f"✅ Found {len(raw_results['documents'][0])} results!")
                        
                        for i, doc in enumerate(raw_results['documents'][0]):
                            with st.expander(f"Result {i+1}"):
                                st.write("**Content:**", doc[:200] + "...")
                                if 'metadatas' in raw_results and raw_results['metadatas'][0]:
                                    st.write("**Metadata:**", raw_results['metadatas'][0][i])
                                if 'distances' in raw_results and raw_results['distances'][0]:
                                    st.write("**Distance:**", f"{raw_results['distances'][0][i]:.3f}")
                    else:
                        st.warning("❌ No results found")
                        
                except Exception as e:
                    st.error(f"Search failed: {e}")
    
    with col2:
        if st.button("Clear Test Data", key="clear_test_data"):
            if st.button("Confirm Clear", key="confirm_clear"):
                # This would need to be implemented based on your ChromaDB setup
                st.warning("Clear function not implemented yet")

# Add this page to your main navigation in streamlit_demo.py
# In the main() function, update the page selection to include:

# page = st.selectbox("Select Page", [
#     "🔍 Search & Query",
#     "📤 Add New Materials", 
#     "📋 Task Management",
#     "📊 Collection Overview",
#     "🐛 Debug Collection",  # Add this
#     "⚙️ Settings"
# ], key="main_page_selector")

# And in the page routing section:
# elif page == "🐛 Debug Collection":
#     debug_collection_page()

if __name__ == "__main__":
    main()
