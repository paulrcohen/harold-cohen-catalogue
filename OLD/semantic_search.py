#!/usr/bin/env python3
"""
Simplified Semantic Search Engine for Harold Cohen Catalogue Raisonné
Focus on reliability over complex diagnostics
"""

import os
import pandas as pd
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import chromadb
from chromadb.config import Settings


class SemanticSearchEngine:
    """Simplified semantic search engine using ChromaDB"""
    
    def __init__(self, collection_name: str = "harold_cohen_docs"):
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.persist_directory = Path("./chroma_data")
        
        self._initialize_chromadb()
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB with simple, reliable approach"""
        try:
            # Ensure directory exists
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            
            # Initialize persistent client with minimal settings
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory)
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                print(f"✅ Loaded existing collection with {self.collection.count()} documents")
            except ValueError:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Harold Cohen catalogue raisonné"}
                )
                print(f"✅ Created new collection: {self.collection_name}")
                # Add some test data if empty
                self._add_initial_test_data()
                
        except Exception as e:
            print(f"❌ ChromaDB initialization failed: {e}")
            # Fall back to in-memory client
            print("🔄 Falling back to in-memory storage")
            self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name
            )
            self._add_initial_test_data()
    
    def _add_initial_test_data(self):
        """Add minimal test data if collection is empty"""
        if self.collection.count() == 0:
            print("Adding initial test data...")
            
            test_docs = [
                "Harold Cohen was a pioneering computer artist who created the AARON drawing program.",
                "The Brooklyn Museum exhibition featured Harold Cohen's computer-generated artwork.",
                "There were shipping delays with the international exhibition materials.",
                "The National Theatre painting was a significant work from Cohen's figurative period.",
                "Question about the National Theater painting - was there an installation issue?"
            ]
            
            test_metadata = [
                {"source": "biography", "type": "text"},
                {"source": "brooklyn_email", "type": "email"},
                {"source": "shipping_email", "type": "email"},
                {"source": "artwork_record", "type": "artwork"},
                {"source": "inquiry_email", "type": "email"}
            ]
            
            test_ids = [f"test_{i}" for i in range(len(test_docs))]
            
            self.collection.add(
                documents=test_docs,
                metadatas=test_metadata,
                ids=test_ids
            )
            print(f"✅ Added {len(test_docs)} test documents")
    
    def get_status(self) -> Dict[str, Any]:
        """Get simple status information"""
        try:
            doc_count = self.collection.count() if self.collection else 0
            is_persistent = isinstance(self.client, chromadb.PersistentClient)
            
            return {
                "status": "ready" if doc_count > 0 else "empty",
                "document_count": doc_count,
                "persistent_storage": is_persistent,
                "storage_path": str(self.persist_directory) if is_persistent else "memory"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "document_count": 0,
                "persistent_storage": False
            }
    
    def ingest_csv_emails(self, csv_path: str, text_column: str = 'message_text') -> Dict[str, Any]:
        """Simple CSV ingestion"""
        try:
            # Read CSV
            df = pd.read_csv(csv_path)
            
            if text_column not in df.columns:
                return {
                    "status": "error", 
                    "message": f"Column '{text_column}' not found in CSV"
                }
            
            # Process documents
            documents = []
            metadatas = []
            ids = []
            
            file_hash = hashlib.md5(str(csv_path).encode()).hexdigest()[:8]
            
            for idx, row in df.iterrows():
                text = str(row.get(text_column, "")).strip()
                if not text or text == "nan":
                    continue
                
                # Simple metadata
                metadata = {
                    "source_file": Path(csv_path).name,
                    "row_index": idx,
                    "ingestion_date": datetime.now().isoformat()[:19]
                }
                
                # Add other relevant columns
                for col in ['sender', 'subject', 'date']:
                    if col in df.columns and not pd.isna(row[col]):
                        metadata[col] = str(row[col])[:200]  # Limit length
                
                documents.append(text)
                metadatas.append(metadata)
                ids.append(f"{file_hash}_{idx}")
            
            if not documents:
                return {"status": "error", "message": "No valid documents found"}
            
            # Add to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            return {
                "status": "success",
                "documents_added": len(documents),
                "total_documents": self.collection.count()
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Simple search function"""
        try:
            if not self.collection:
                return []
            
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
                        'id': ids[i] if i < len(ids) else f"result_{i}"
                    })
            
            return formatted
            
        except Exception as e:
            print(f"Search error: {e}")
            return []


def test_engine():
    """Simple test function"""
    engine = SemanticSearchEngine()
    status = engine.get_status()
    print("Status:", status)
    
    results = engine.search("Brooklyn museum")
    print(f"Search results: {len(results)}")
    for r in results[:2]:
        print(f"- {r['content'][:80]}...")


if __name__ == "__main__":
    test_engine()
