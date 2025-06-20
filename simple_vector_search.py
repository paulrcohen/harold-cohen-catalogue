#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 12:38:07 2025

@author: prcohen
"""

#!/usr/bin/env python3
"""
Simple vector search engine that works everywhere
No ChromaDB dependencies - uses sentence-transformers directly
"""

import streamlit as st
import pandas as pd
import pickle
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
import hashlib

# Only import sentence-transformers - much more reliable than ChromaDB
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    transformers_available = True
except ImportError:
    transformers_available = False


class SimpleVectorSearch:
    """Simple vector search without ChromaDB dependencies"""
    
    def __init__(self):
        self.documents = []
        self.metadatas = []
        self.embeddings = None
        self.model = None
        self.backup_file = "simple_search_backup.pkl"
        self._initialize()
    
    def _initialize(self):
        """Initialize the search engine"""
        try:
            if transformers_available:
                # Use a lightweight, fast model
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                st.success("✅ Vector search engine initialized")
            else:
                st.error("❌ sentence-transformers not available - using basic text search")
                return
            
            # Try to restore from backup
            self._restore_from_backup()
            
            # Add initial data if empty
            if len(self.documents) == 0:
                self._add_initial_data()
                
        except Exception as e:
            st.error(f"Search engine initialization failed: {e}")
            # Fallback to basic search
            self.model = None
    
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
        
        self.add_documents(test_docs, test_metadata, "initial_data")
    
    def _save_backup(self):
        """Save all data to backup file"""
        try:
            backup_data = {
                'documents': self.documents,
                'metadatas': self.metadatas,
                'embeddings': self.embeddings.tolist() if self.embeddings is not None else None,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.backup_file, 'wb') as f:
                pickle.dump(backup_data, f)
            
            print(f"💾 Saved backup with {len(self.documents)} documents")
                
        except Exception as e:
            print(f"❌ Backup save failed: {e}")
    
    def _restore_from_backup(self):
        """Restore from backup file"""
        try:
            from pathlib import Path
            if Path(self.backup_file).exists():
                print(f"Found backup file: {self.backup_file}")
                
                with open(self.backup_file, 'rb') as f:
                    backup_data = pickle.load(f)
                
                self.documents = backup_data.get('documents', [])
                self.metadatas = backup_data.get('metadatas', [])
                
                embeddings_data = backup_data.get('embeddings')
                if embeddings_data:
                    self.embeddings = np.array(embeddings_data)
                
                backup_timestamp = backup_data.get('timestamp', 'unknown')
                print(f"✅ Restored {len(self.documents)} documents from backup (created: {backup_timestamp})")
            else:
                print(f"No backup file found at: {self.backup_file}")
                
        except Exception as e:
            print(f"❌ Backup restore failed: {e}")
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None, source_name: str = "unknown") -> int:
        """Add documents to the collection"""
        try:
            if not documents:
                return 0
            
            # Filter valid documents
            valid_docs = []
            valid_metas = []
            
            for i, doc in enumerate(documents):
                if doc and len(str(doc).strip()) > 10:
                    valid_docs.append(str(doc).strip())
                    
                    if metadatas and i < len(metadatas):
                        valid_metas.append(metadatas[i])
                    else:
                        valid_metas.append({
                            "source": source_name,
                            "type": "uploaded",
                            "date": datetime.now().isoformat()[:10]
                        })
            
            if not valid_docs:
                return 0
            
            # Add to our storage
            self.documents.extend(valid_docs)
            self.metadatas.extend(valid_metas)
            
            # Generate embeddings if model available
            if self.model:
                new_embeddings = self.model.encode(valid_docs)
                
                if self.embeddings is None:
                    self.embeddings = new_embeddings
                else:
                    self.embeddings = np.vstack([self.embeddings, new_embeddings])
            
            # Save backup
            self._save_backup()
            
            return len(valid_docs)
            
        except Exception as e:
            print(f"Add documents error: {e}")
            return 0
    
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search the collection"""
        try:
            if not self.documents:
                return []
            
            if self.model and self.embeddings is not None:
                # Vector search
                query_embedding = self.model.encode([query])
                similarities = cosine_similarity(query_embedding, self.embeddings)[0]
                
                # Get top results
                top_indices = np.argsort(similarities)[::-1][:n_results]
                
                results = []
                for idx in top_indices:
                    if idx < len(self.documents):
                        results.append({
                            'content': self.documents[idx],
                            'metadata': self.metadatas[idx] if idx < len(self.metadatas) else {},
                            'relevance': float(similarities[idx]),
                            'distance': 1 - float(similarities[idx]),
                            'id': f"doc_{idx}"
                        })
                
                return results
            else:
                # Fallback: basic text search
                query_lower = query.lower()
                matches = []
                
                for i, doc in enumerate(self.documents):
                    if query_lower in doc.lower():
                        score = doc.lower().count(query_lower) / len(doc.split())
                        matches.append({
                            'content': doc,
                            'metadata': self.metadatas[i] if i < len(self.metadatas) else {},
                            'relevance': score,
                            'distance': 1 - score,
                            'id': f"doc_{i}"
                        })
                
                # Sort by relevance
                matches.sort(key=lambda x: x['relevance'], reverse=True)
                return matches[:n_results]
                
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def ingest_csv(self, df: pd.DataFrame, text_column: str, source_name: str = "csv_upload") -> Dict[str, Any]:
        """Ingest documents from CSV DataFrame"""
        try:
            documents = []
            metadatas = []
            
            for idx, row in df.iterrows():
                text = str(row.get(text_column, "")).strip()
                if text and text != "nan" and len(text) > 10:
                    documents.append(text)
                    
                    # Create metadata
                    metadata = {
                        "source": source_name,
                        "row_index": idx,
                        "type": "email",
                        "date": datetime.now().isoformat()[:10]
                    }
                    
                    # Add other columns
                    for col in ['sender', 'subject', 'date', 'recipient']:
                        if col in df.columns and not pd.isna(row[col]):
                            metadata[col] = str(row[col])[:200]
                    
                    metadatas.append(metadata)
            
            if documents:
                added = self.add_documents(documents, metadatas, source_name)
                return {
                    "status": "success",
                    "documents_added": added,
                    "total_documents": len(self.documents)
                }
            else:
                return {"status": "error", "message": "No valid documents found"}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            from pathlib import Path
            backup_exists = Path(self.backup_file).exists()
            
            return {
                "document_count": len(self.documents),
                "backup_exists": backup_exists,
                "vector_search": self.model is not None,
                "status": "ready" if self.documents else "empty"
            }
        except Exception as e:
            return {"status": "error", "error": str(e), "document_count": 0}


# Streamlit cache for the search engine
@st.cache_resource
def get_search_engine():
    """Get cached search engine instance"""
    return SimpleVectorSearch()
