#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 20:12:37 2025

@author: prcohen
"""

#!/usr/bin/env python3
"""
Modular Catalogue Raisonné System for Harold Cohen
Orchestrator module
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

# Other modules
from email_text_processing import EmailTextProcessor 


# ============================================================================
# 5. MAIN ORCHESTRATOR CLASS
# ============================================================================

class CatalogueRaisonneSystem:
    """
    Main orchestrator that combines all modules
    """
    
    def __init__(self, 
                 email_csv_path: Optional[str] = None,
                 artwork_csv_path: Optional[str] = None,
                 anthropic_api_key: Optional[str] = None):
        
        # Initialize modules
        self.email_processor = EmailTextProcessor(email_csv_path) if email_csv_path else None
        self.semantic_search = SemanticSearchEngine()
        self.artwork_db = ArtworkDatabase()
        self.response_generator = ResponseGenerator(anthropic_api_key)
        
        # Load data if paths provided
        if artwork_csv_path:
            self.artwork_db.load_from_csv(artwork_csv_path)
        
        if email_csv_path:
            self.semantic_search.ingest_csv_emails(email_csv_path)
    
    def comprehensive_search(self, query: str) -> Dict[str, Any]:
        """
        Perform comprehensive search across all modules
        """
        results = {
            'query': query,
            'timestamp': datetime.now(),
            'semantic_results': [],
            'artwork_matches': [],
            'email_matches': pd.DataFrame(),
            'response': ''
        }
        
        # 1. Semantic search
        semantic_results = self.semantic_search.search(query, n_results=5)
        results['semantic_results'] = semantic_results
        
        # 2. Search for artwork references in the query
        artwork_ids = self._extract_artwork_ids_from_query(query)
        artwork_matches = []
        for artwork_id in artwork_ids:
            artwork = self.artwork_db.get_artwork(artwork_id)
            if artwork:
                artwork_matches.append(artwork)
        results['artwork_matches'] = artwork_matches
        
        # 3. Search emails if processor available
        if self.email_processor:
            # Try boolean search with query terms
            query_terms = query.lower().split()
            # Filter out common words
            meaningful_terms = [term for term in query_terms if len(term) > 3]
            if meaningful_terms:
                email_results = self.email_processor.boolean_search(meaningful_terms[:3], 'OR')
                results['email_matches'] = email_results
        
        # 4. Generate response
        response = self.response_generator.generate_response(
            query, 
            semantic_results, 
            artwork_matches
        )
        results['response'] = response
        
        return results
    
    def _extract_artwork_ids_from_query(self, query: str) -> List[str]:
        """Extract artwork IDs from query text"""
        patterns = [
            r'HC\d{4}',
            r'i23-\d{4}',
            r'HC-\d{4}',
            r'I23-\d{4}',
        ]
        
        found_ids = []
        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            found_ids.extend(matches)
        
        return list(set(found_ids))  # Remove duplicates

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """Example of how to use the modular system"""
    
    # Initialize individual modules
    print("=== Testing Email Text Processor ===")
    email_processor = EmailTextProcessor("./emails/artwork_mentions.csv")
    
    # Find artwork references
    artwork_refs = email_processor.find_artwork_references()
    print(f"Found {len(artwork_refs)} emails with artwork references")
    
    # Get summary
    summary = email_processor.get_artwork_mention_summary()
    print(f"Summary: {summary}")
    
    print("\n=== Testing Semantic Search ===")
    semantic_engine = SemanticSearchEngine()
    
    # Ingest emails
    stats = semantic_engine.ingest_csv_emails("./emails/artwork_mentions.csv")
    print(f"Ingestion stats: {stats}")
    
    # Search
    results = semantic_engine.search("painting machine")
    print(f"Found {len(results)} semantic matches")
    
    print("\n=== Testing Full System ===")
    # Initialize full system
    system = CatalogueRaisonneSystem(
        email_csv_path="./emails/artwork_mentions.csv"
    )
    
    # Comprehensive search
    results = system.comprehensive_search("Tell me about the painting machine")
    print(f"Response: {results['response'][:200]}...")

if __name__ == "__main__":
    example_usage()
