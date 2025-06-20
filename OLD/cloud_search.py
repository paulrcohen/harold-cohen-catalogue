#!/usr/bin/env python3
"""
Cloud-ready search engine with simple, reliable persistence
"""

import streamlit as st
import chromadb
import pandas as pd
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


class CloudSearchEngine:
    """Simple, cloud-compatible search engine"""
    
    def __init__(self):
        self.collection = None
        self.backup_file = "collection_backup.pkl"
        self._initialize()
    
    def _initialize(self):
        """Initialize ChromaDB with backup/restore capability"""
        try:
            # Always start with in-memory client (cloud-compatible)
            client = chromadb.Client()
            self.collection = client.get_or_create_collection("harold_cohen")
            
            # Try to restore from backup
            self._restore_from_backup()
            
            # Add initial test data if empty
            if self.collection.count() == 0:
                self._add_initial_data()
                
        except Exception as e:
            st.error(f"Search engine initialization failed: {e}")
            raise
    
    def _add_initial_data(self):
        """Add initial test data"""
        test_docs = [
            "Harold Cohen was a pioneering computer artist who created the AARON drawing program from the 1970s onwards.",
            "The Brooklyn Museum exhibition featured Harold Cohen's computer-generated artwork and large-scale prints.",
            "There were shipping delays with the international exhibition materials due to customs paperwork.",
            "The National Theatre painting was a significant 6x4 foot work from Cohen's figurative period.",
            "Installation issues occurred with the National Theater painting - there were problems with the frame."
        ]
        
        test_metadata = [
            {"source": "biography", "type": "reference", "date": "2025-01-01"},
            {"source": "brooklyn_exhibition", "type": "email", "date": "2024-12-15"},
            {"source": "shipping_update", "type": "email", "date": "2024-12-20"},
            {"source": "artwork_record", "type": "catalog", "date": "2024-11-10"},
            {"source": "installation_notes", "type": "email", "date": "2024-12-22"}
        ]
        
        self.collection.add(
            documents=test_docs,
            metadatas=test_metadata,
            ids=[f"initial_{i}" for i in range(len(test_docs))]
        )
        
        # Save backup immediately
        self._save_backup()
    
    def _save_backup(self):
        """Save collection data to backup file"""
        try:
            if self.collection.count() > 0:
                data = self.collection.get()
                backup_data = {
                    'documents': data.get('documents', []),
                    'metadatas': data.get('metadatas', []),
                    'ids': data.get('ids', []),
                    'timestamp': datetime.now().isoformat()
                }
                
                with open(self.backup_file, 'wb') as f:
                    pickle.dump(backup_data, f)
                
                print(f"💾 Saved backup with {len(backup_data['documents'])} documents to {self.backup_file}")
                    
        except Exception as e:
            print(f"❌ Backup save failed: {e}")
    
    def _restore_from_backup(self):
        """Restore collection from backup file"""
        try:
            if Path(self.backup_file).exists():
                print(f"Found backup file: {self.backup_file}")
                with open(self.backup_file, 'rb') as f:
                    backup_data = pickle.load(f)
                
                if backup_data.get('documents'):
                    self.collection.add(
                        documents=backup_data['documents'],
                        metadatas=backup_data['metadatas'],
                        ids=backup_data['ids']
                    )
                    backup_timestamp = backup_data.get('timestamp', 'unknown')
                    print(f"✅ Restored {len(backup_data['documents'])} documents from backup (created: {backup_timestamp})")
                else:
                    print("❌ Backup file exists but contains no documents")
            else:
                print(f"No backup file found at: {self.backup_file}")
                    
        except Exception as e:
            print(f"❌ Backup restore failed: {e}")
    
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search the collection"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Format results
            formatted = []
            if results.get('documents') and results['documents'][0]:
                docs = results['documents'][0]
                metadatas = results.get('metadatas', [[]])[0]
                distances = results.get('distances', [[]])[0]
                ids = results.get('ids', [[]])[0]
                
                for i in range(len(docs)):
                    formatted.append({
                        'content': docs[i],
                        'metadata': metadatas[i] if i < len(metadatas) else {},
                        'distance': distances[i] if i < len(distances) else 0,
                        'relevance': 1 - (distances[i] if i < len(distances) else 0),
                        'id': ids[i] if i < len(ids) else f"result_{i}"
                    })
            
            return formatted
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None, source_name: str = "unknown") -> int:
        """Add documents to collection"""
        try:
            if not documents:
                return 0
            
            # Generate IDs and metadata
            base_id = hashlib.md5(source_name.encode()).hexdigest()[:8]
            ids = [f"{base_id}_{i}" for i in range(len(documents))]
            
            if metadatas is None:
                metadatas = [{"source": source_name, "type": "uploaded", "date": datetime.now().isoformat()[:10]} 
                           for _ in documents]
            
            # Add to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            # Save backup
            self._save_backup()
            
            return len(documents)
            
        except Exception as e:
            print(f"Add documents error: {e}")
            return 0
    
    def ingest_csv(self, df: pd.DataFrame, text_column: str, source_name: str = "csv_upload") -> Dict[str, Any]:
        """Ingest documents from CSV DataFrame"""
        try:
            documents = []
            metadatas = []
            
            for idx, row in df.iterrows():
                text = str(row.get(text_column, "")).strip()
                if text and text != "nan" and len(text) > 10:  # Basic filtering
                    documents.append(text)
                    
                    # Create metadata from other columns
                    metadata = {
                        "source": source_name,
                        "row_index": idx,
                        "type": "email",
                        "date": datetime.now().isoformat()[:10]
                    }
                    
                    # Add other relevant columns
                    for col in ['sender', 'subject', 'date', 'recipient']:
                        if col in df.columns and not pd.isna(row[col]):
                            metadata[col] = str(row[col])[:200]  # Limit length
                    
                    metadatas.append(metadata)
            
            if documents:
                added = self.add_documents(documents, metadatas, source_name)
                return {
                    "status": "success",
                    "documents_added": added,
                    "total_documents": self.collection.count()
                }
            else:
                return {"status": "error", "message": "No valid documents found"}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            count = self.collection.count()
            backup_exists = Path(self.backup_file).exists()
            
            return {
                "document_count": count,
                "backup_exists": backup_exists,
                "status": "ready" if count > 0 else "empty"
            }
        except Exception as e:
            return {"status": "error", "error": str(e), "document_count": 0}


# Streamlit cache for the search engine
@st.cache_resource
def get_search_engine():
    """Get cached search engine instance"""
    return CloudSearchEngine()
