#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 20:48:02 2025

@author: prcohen
"""

#!/usr/bin/env python3
"""
Modular Catalogue Raisonné System for Harold Cohen
Response generation module
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


# Get API key from environment or Streamlit secrets
api_key = os.getenv('ANTHROPIC_API_KEY')
if not api_key and 'ANTHROPIC_API_KEY' in st.secrets:
    api_key = st.secrets['ANTHROPIC_API_KEY']
if api_key:
    os.environ['ANTHROPIC_API_KEY'] = api_key

class ResponseGenerator:
    """
    Module for generating responses using retrieved context
    """
    
    def __init__(self, anthropic_api_key: Optional[str] = None):
        # Get API key from environment variable if not provided
        api_key = anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')
        self.anthropic_client = Anthropic(api_key=api_key) if api_key else None
    
    def format_search_results(self, raw_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert ChromaDB query results to the expected format
        
        Args:
            raw_results: Raw results from collection.query()
            
        Returns:
            List of formatted results
        """
        formatted_results = []
        
        if 'documents' in raw_results and raw_results['documents']:
            documents = raw_results['documents'][0]  # First query result
            metadatas = raw_results.get('metadatas', [[]])[0]
            distances = raw_results.get('distances', [[]])[0]
            ids = raw_results.get('ids', [[]])[0]
            
            for i in range(len(documents)):
                formatted_results.append({
                    'content': documents[i],
                    'metadata': metadatas[i] if i < len(metadatas) else {},
                    'distance': distances[i] if i < len(distances) else None,
                    'id': ids[i] if i < len(ids) else f"result_{i}"
                })
        
        return formatted_results
    
    def generate_response(self, 
                         query: str, 
                         context_chunks: List[Dict[str, Any]],
                         artwork_context: Optional[List] = None,
                         max_chunks: int = 3,
                         max_chars_per_chunk: int = 1000,
                         use_cheaper_model: bool = False) -> str:
        """Generate response using Claude with retrieved context"""
        
        if not self.anthropic_client:
            return self._generate_simple_response(query, context_chunks, artwork_context)
        
        # Limit and truncate chunks to control costs
        limited_chunks = context_chunks[:max_chunks]
        
        # Build context from retrieved chunks with length limits
        context_parts = []
        total_chars = 0
        
        for i, chunk in enumerate(limited_chunks, 1):
            source = chunk['metadata'].get('source_file', 'Unknown')
            source_type = chunk['metadata'].get('source_type', 'Unknown')
            content = chunk['content']
            
            # Truncate content if too long
            if len(content) > max_chars_per_chunk:
                content = content[:max_chars_per_chunk] + "... [truncated]"
            
            context_part = (
                f"Source {i}: {source} ({source_type})\n"
                f"Content: {content}\n"
            )
            
            # Stop if we're getting too much context
            if total_chars + len(context_part) > 5000:  # ~5000 chars ≈ 1250 tokens
                context_parts.append(f"[Additional sources truncated to control costs]")
                break
                
            context_parts.append(context_part)
            total_chars += len(context_part)
        
        context_text = "\n".join(context_parts)
        
        # Add artwork context if available
        artwork_text = ""
        if artwork_context:
            artwork_text = "\n\nRelated Artworks:\n" + "\n".join([
                f"- {artwork.id}: {artwork.title} ({artwork.year or 'Unknown year'})"
                for artwork in artwork_context
            ])
        
        prompt = f"""You are a research assistant helping with Harold Cohen's catalogue raisonné, focusing on his figurative period (early 1980s - late 1990s). 

You have access to various sources including emails, publications, and documentation about Cohen's work. Your role is to:
1. Answer questions based on the provided sources
2. Always cite your sources clearly
3. If the source is email, say who it is from and to
4. Distinguish between documented facts and reasonable inferences
5. Flag when information is missing or uncertain
6. Focus on being helpful while maintaining scholarly rigor

User Question: {query}

Retrieved Context:
{context_text}{artwork_text}

Please provide a helpful response based on the available context. If you make any inferences beyond what's directly stated, please make that clear. Always cite the specific sources you're drawing from."""

        # Choose model based on cost preference
        model_name = "claude-3-haiku-20240307" if use_cheaper_model else "claude-3-5-sonnet-20241022"
        
        try:
            response = self.anthropic_client.messages.create(
                model=model_name,
                max_tokens=500 if use_cheaper_model else 1000,  # Fewer tokens for cheaper model
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _generate_simple_response(self, 
                                 query: str, 
                                 context_chunks: List[Dict[str, Any]],
                                 artwork_context: Optional[List] = None) -> str:
        """Generate a simple response without Claude API"""
        response_parts = [
            f"Query: {query}",
            f"\nFound {len(context_chunks)} relevant text passages:\n"
        ]
        
        for i, chunk in enumerate(context_chunks[:3], 1):  # Show top 3
            source = chunk['metadata'].get('source_file', 'Unknown')
            content_preview = chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content']
            response_parts.append(f"{i}. Source: {source}\n   Content: {content_preview}\n")
        
        if artwork_context:
            response_parts.append(f"\nRelated artworks found: {len(artwork_context)}")
            for artwork in artwork_context[:5]:  # Show top 5
                response_parts.append(f"- {artwork.id}: {artwork.title}")
        
        return "\n".join(response_parts)


def example_rag_usage():
    """Complete example of RAG system with cost optimization"""
    print("\n=== RAG System Example (Cost Optimized) ===")
    
    # Import and initialize semantic search
    from semantic_search import SemanticSearchEngine
    
    # Initialize components
    search_engine = SemanticSearchEngine()
    response_generator = ResponseGenerator()
    
    # Check what's in the collection
    stats = search_engine.get_collection_stats()
    print(f"Collection has {stats.get('total_documents', 0)} documents")
    
    # Perform search with limited results
    query = "How many posters from the Brooklyn museum did we ship?"
    print(f"\nSearching for: '{query}'")
    
    # Get raw results from ChromaDB (limit to 2-3 for cost control)
    raw_results = search_engine.collection.query(
        query_texts=[query],
        n_results=2  # Reduced from 3
    )
    
    # Format results for response generation
    formatted_results = response_generator.format_search_results(raw_results)
    print(f"Found {len(formatted_results)} relevant passages")
    
    # Show what we're sending to Claude
    total_chars = sum(len(chunk['content']) for chunk in formatted_results)
    print(f"Total context size: ~{total_chars} characters (~{total_chars//4} tokens)")
    
    # Generate response with cost controls
    print("\n--- RAG Response (Using Cheaper Haiku Model) ---")
    response = response_generator.generate_response(
        query=query,
        context_chunks=formatted_results,
        artwork_context=None,
        max_chunks=5,  # Only use top 5 results
        max_chars_per_chunk=800,  # Limit chunk size
        use_cheaper_model=True  # Use Haiku instead of Sonnet
    )
    print(response)
    
    return search_engine, response_generator


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Run the complete example
    search_engine, rag = example_rag_usage()
    
    # Additional test queries
    test_queries = [

        "Any shipping problems or delays?",
        "Any problems with customs?"
    ]
    
    print("\n=== Additional Test Queries ===")
    for query in test_queries:
        print(f"\n--- Query: {query} ---")
        
        # Search
        raw_results = search_engine.collection.query(
            query_texts=[query],
            n_results=2
        )
        
        # Format and generate response
        formatted_results = rag.format_search_results(raw_results)
        response = rag.generate_response(
            query=query,
            context_chunks=formatted_results
        )
        print(response[:2000] + "..." if len(response) > 2000 else response)
