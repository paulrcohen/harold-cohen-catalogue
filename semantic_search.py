#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 15:57:57 2025

@author: prcohen
"""

#!/usr/bin/env python3
"""
Modular Catalogue Raisonné System for Harold Cohen
Semantic search module with persistence checking
"""

import os
import re
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib

# Core dependencies
import chromadb
from chromadb.config import Settings
from anthropic import Anthropic


class SemanticSearchEngine:
    """
    ChromaDB-based semantic search engine for text corpus with persistence checking
    """
    
    def __init__(self, collection_name: str = "cohen_catalog", persist_directory: str = "./chromadb"):
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB with explicit settings
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Try to get existing collection first
            try:
                self.collection = self.client.get_collection(name=collection_name)
                print(f"Loaded existing collection: {collection_name}")
            except:
                # Create new collection if it doesn't exist
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"description": "Harold Cohen catalogue raisonné corpus"}
                )
                print(f"Created new collection: {collection_name}")
                
        except Exception as e:
            print(f"ChromaDB initialization error: {e}")
            print("Trying alternative initialization...")
            
            # Fallback: try with in-memory client first
            self.client = chromadb.Client()
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Harold Cohen catalogue raisonné corpus"}
            )
            print("Using in-memory ChromaDB (data won't persist)")
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate MD5 hash of a file for change detection"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get file metadata for tracking"""
        file_path = Path(file_path)
        return {
            'file_path': str(file_path.absolute()),
            'file_name': file_path.name,
            'file_size': file_path.stat().st_size,
            'file_hash': self._get_file_hash(file_path),
            'modified_time': file_path.stat().st_mtime,
            'ingested_at': datetime.now().isoformat()
        }
    
    def is_corpus_ingested(self, csv_file_path: str, check_for_changes: bool = True) -> Tuple[bool, Optional[Dict]]:
        """
        Check if a corpus has already been ingested
        
        Args:
            csv_file_path: Path to the CSV file
            check_for_changes: If True, also check if the file has changed since ingestion
            
        Returns:
            Tuple of (is_ingested, existing_info)
        """
        try:
            file_path = Path(csv_file_path)
            absolute_path = str(file_path.absolute())
            relative_path = str(file_path)
            filename = file_path.name
            
            # Check for documents with absolute path first
            results = self.collection.get(
                where={"source_file": absolute_path},
                limit=1
            )
            
            # If not found, check relative path
            if not results['ids']:
                results = self.collection.get(
                    where={"source_file": relative_path},
                    limit=1
                )
            
            # If still not found, check by filename
            if not results['ids']:
                results = self.collection.get(
                    where={"file_name": filename},
                    limit=1
                )
            
            if not results['ids']:
                return False, None
            
            # Get the ingestion metadata from the first document
            existing_metadata = results['metadatas'][0] if results['metadatas'] else {}
            
            if not check_for_changes:
                return True, existing_metadata
            
            # Check if file has changed since ingestion
            current_file_info = self._get_file_info(csv_file_path)
            
            # Compare file hash if available
            if 'file_hash' in existing_metadata:
                if existing_metadata['file_hash'] != current_file_info['file_hash']:
                    return False, existing_metadata  # File has changed
            
            # Compare modification time as fallback
            elif 'modified_time' in existing_metadata:
                if existing_metadata['modified_time'] != current_file_info['modified_time']:
                    return False, existing_metadata  # File has changed
            
            return True, existing_metadata
            
        except Exception as e:
            print(f"Error checking corpus ingestion status: {e}")
            return False, None
    
    def remove_corpus(self, csv_file_path: str) -> Dict[str, Any]:
        """
        Remove all documents from a specific corpus file
        
        Args:
            csv_file_path: Path to the CSV file whose documents should be removed
            
        Returns:
            Statistics about removal
        """
        try:
            file_path = Path(csv_file_path)
            absolute_path = str(file_path.absolute())
            relative_path = str(file_path)
            filename = file_path.name
            
            # Find documents with absolute path
            results_abs = self.collection.get(
                where={"source_file": absolute_path}
            )
            
            # Find documents with relative path
            results_rel = self.collection.get(
                where={"source_file": relative_path}
            )
            
            # Find documents with just filename (in case there are other variations)
            results_name = self.collection.get(
                where={"file_name": filename}
            )
            
            # Combine all IDs and remove duplicates
            all_ids = set()
            if results_abs['ids']:
                all_ids.update(results_abs['ids'])
            if results_rel['ids']:
                all_ids.update(results_rel['ids'])
            if results_name['ids']:
                all_ids.update(results_name['ids'])
            
            if not all_ids:
                return {'removed_count': 0, 'message': 'No documents found for this corpus'}
            
            # Delete all documents
            self.collection.delete(ids=list(all_ids))
            
            return {
                'removed_count': len(all_ids),
                'message': f"Removed {len(all_ids)} documents from corpus (including duplicates)"
            }
            
        except Exception as e:
            return {'error': str(e), 'removed_count': 0}
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            chunks.append(chunk_text)
        
        return chunks
    
    def ingest_csv_emails(self, 
                         csv_file_path: str, 
                         text_column: str = 'message_text',
                         chunk_size: int = 1000,
                         force_reingest: bool = False) -> Dict[str, Any]:
        """
        Ingest emails from CSV into semantic search index
        
        Args:
            csv_file_path: Path to CSV file
            text_column: Column containing text to index
            chunk_size: Size of text chunks
            force_reingest: If True, re-ingest even if already present
            
        Returns:
            Statistics about ingestion
        """
        
        # Check if corpus is already ingested
        is_ingested, existing_info = self.is_corpus_ingested(csv_file_path)
        
        if is_ingested and not force_reingest:
            return {
                'status': 'already_ingested',
                'message': f'Corpus already ingested. Use force_reingest=True to re-ingest.',
                'existing_info': existing_info,
                'processed_emails': 0,
                'total_chunks': 0
            }
        
        # If force_reingest or file has changed, remove existing documents
        if force_reingest or (existing_info is not None):
            removal_stats = self.remove_corpus(csv_file_path)
            print(f"Removed existing corpus: {removal_stats}")
        
        # Get file information for tracking
        file_info = self._get_file_info(csv_file_path)
        
        # Proceed with ingestion
        df = pd.read_csv(csv_file_path)
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in CSV")
        
        total_chunks = 0
        processed_emails = 0
        
        for index, row in df.iterrows():
            message_text = str(row[text_column]) if pd.notna(row[text_column]) else ""
            
            if not message_text.strip():
                continue
            
            # Create chunks
            chunks = self.chunk_text(message_text, chunk_size)
            
            # Prepare metadata including file tracking info
            base_metadata = {
                'source_type': 'email',
                'source_file': file_info['file_path'],
                'file_name': file_info['file_name'],
                'file_hash': file_info['file_hash'],
                'file_size': file_info['file_size'],
                'modified_time': file_info['modified_time'],
                'ingested_at': file_info['ingested_at'],
                'email_index': index,
                'total_chunks': len(chunks)
            }
            
            # Add other CSV columns as metadata
            for col in df.columns:
                if col != text_column and pd.notna(row[col]):
                    base_metadata[col] = str(row[col])
            
            # Add chunks to collection
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"email_{index}_chunk_{chunk_idx}_{file_info['file_hash'][:8]}"
                
                chunk_metadata = {
                    **base_metadata,
                    'chunk_index': chunk_idx,
                    'chunk_id': chunk_id
                }
                
                self.collection.add(
                    documents=[chunk],
                    metadatas=[chunk_metadata],
                    ids=[chunk_id]
                )
                
                total_chunks += 1
            
            processed_emails += 1
        
        return {
            'status': 'ingested',
            'processed_emails': processed_emails,
            'total_chunks': total_chunks,
            'avg_chunks_per_email': total_chunks / processed_emails if processed_emails > 0 else 0,
            'file_info': file_info
        }
    
    def search(self, 
               query: str, 
               n_results: int = 10,
               filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Perform semantic search
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
        """
        search_kwargs = {
            'query_texts': [query],
            'n_results': n_results
        }
        
        if filter_metadata:
            search_kwargs['where'] = filter_metadata
        
        results = self.collection.query(**search_kwargs)
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None,
                'id': results['ids'][0][i]
            })
        
        return formatted_results
    
    def search_emails_only(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Search only within email content"""
        return self.search(query, n_results, filter_metadata={'source_type': 'email'})
    
    def search_corpus(self, query: str, csv_file_path: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Search within a specific corpus file"""
        return self.search(
            query, 
            n_results, 
            filter_metadata={
                'source_type': 'email',
                'source_file': str(Path(csv_file_path).absolute())
            }
        )
    
    def list_ingested_corpora(self) -> List[Dict[str, Any]]:
        """List all ingested corpora with their metadata"""
        try:
            # Get unique source files
            results = self.collection.get()
            
            corpora = {}
            for metadata in results['metadatas']:
                if 'source_file' in metadata:
                    file_path = metadata['source_file']
                    if file_path not in corpora:
                        corpora[file_path] = {
                            'file_path': file_path,
                            'file_name': metadata.get('file_name', 'unknown'),
                            'ingested_at': metadata.get('ingested_at', 'unknown'),
                            'file_hash': metadata.get('file_hash', 'unknown'),
                            'document_count': 0
                        }
                    corpora[file_path]['document_count'] += 1
            
            return list(corpora.values())
            
        except Exception as e:
            return [{'error': str(e)}]
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count_result = self.collection.count()
            
            # Get sample of metadata to understand structure
            sample = self.collection.get(limit=10)
            
            metadata_fields = set()
            source_types = set()
            
            for metadata in sample['metadatas']:
                metadata_fields.update(metadata.keys())
                if 'source_type' in metadata:
                    source_types.add(metadata['source_type'])
            
            return {
                'total_documents': count_result,
                'metadata_fields': sorted(list(metadata_fields)),
                'source_types': sorted(list(source_types)),
                'sample_metadata': sample['metadatas'][0] if sample['metadatas'] else {},
                'ingested_corpora': self.list_ingested_corpora()
            }
        except Exception as e:
            return {'error': str(e)}


def example_usage(mail_corpus_filepath):
    """Enhanced example showing persistence checking"""
    print("\n=== Testing Semantic Search with Persistence ===")
    semantic_engine = SemanticSearchEngine()
    
    # Check current state
    print("\n--- Current Collection Stats ---")
    stats = semantic_engine.get_collection_stats()
    print(f"Total documents: {stats.get('total_documents', 0)}")
    print(f"Ingested corpora: {len(stats.get('ingested_corpora', []))}")
    
    # Check if corpus is already ingested
    csv_path = mail_corpus_filepath
    is_ingested, existing_info = semantic_engine.is_corpus_ingested(csv_path)
    
    print(f"\n--- Corpus Status ---")
    print(f"Corpus '{csv_path}' already ingested: {is_ingested}")
    if existing_info:
        print(f"Previously ingested at: {existing_info.get('ingested_at', 'unknown')}")
    
    # Ingest only if needed (or force with force_reingest=True)
    print(f"\n--- Ingestion ---")
    if os.path.exists(csv_path):
        ingestion_stats = semantic_engine.ingest_csv_emails(
            csv_path, 
            force_reingest=False  # Change to True to force re-ingestion
        )
        print(f"Ingestion result: {ingestion_stats['status']}")
        if ingestion_stats['status'] == 'ingested':
            print(f"Processed {ingestion_stats['processed_emails']} emails")
            print(f"Created {ingestion_stats['total_chunks']} chunks")
    else:
        print(f"CSV file not found: {csv_path}")
    
    # Search
    print(f"\n--- Search ---")
    results = semantic_engine.search("painting machine", n_results=3)
    print(f"Found {len(results)} semantic matches")
    for i, result in enumerate(results):
        print(f"Result {i+1}: {result['content'][:100]}...")
    
    return semantic_engine


if __name__ == "__main__":
    se = example_usage("./machnik_emails_short")

    # Example of manual search after setup
    print("\n=== Manual Search Example ===")
    results = se.collection.query(
        query_texts=["Did we ship paintings from Encinitas to London in 2023?"],
        n_results=2
    )
    print(f"Manual search results: {len(results['documents'][0])} matches found")

# results = se.collection.query(
#     query_texts=["How many posters from the Brooklyn museum did we ship?"], # Chroma will embed this for you
#     n_results=2 # how many results to return
# )
#%%
# # To remove a corpus (perhaps one of several from a SemanticSearchEngine object
# removal_stats = se.remove_corpus("./machnik_emails.csv")
# print(removal_stats)

# # To re-ingest a corpus into a SemanticSearchEngine object

# ingestion_stats = se.ingest_csv_emails("./machnik_emails.csv", force_reingest=True)
# print(ingestion_stats)

# To completely remove all chroma databases and start over.  Need to restart python console as well.
# import shutil
# import os

# # Remove ChromaDB directory completely
# if os.path.exists("./chromadb"):
#     shutil.rmtree("./chromadb")

# # Also try removing any hidden ChromaDB files
# for item in os.listdir("."):
#     if item.startswith(".chroma"):
#         if os.path.isdir(item):
#             shutil.rmtree(item)
#         else:
#             os.remove(item)
#%%
# for k,v in results.items():
#     print(f"===================== {k} ===================\n{v}")
#     if k == 'metadatas':
#         for m in v:
#             for md in m:
#              print(f"\n\tMetadata: {type(md)} {sorted(md.items(),key = lambda x: x[0])}\n")
#     elif k == 'documents':
#         for d in v:
#             print(type(d))
#             for doc in d:
#                 print(f"\n\tDocument: {doc}")
