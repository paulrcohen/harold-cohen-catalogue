#!/usr/bin/env python3
"""
Modular Catalogue Raisonné System for Harold Cohen
Email text processing module
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

class EmailTextProcessor:
    """
    Module for processing extracted email CSV files with search capabilities
    """
    
    def __init__(self, csv_file_path: str):
        self.csv_file_path = csv_file_path
        self.df = pd.read_csv(csv_file_path)
        self.artwork_id_patterns = [
            r'HC\d{4}',           # HCxxxx format
            r'i23-\d{4}',         # i23-xxxx format
            r'HC-\d{4}',          # HC-xxxx format (variant)
            r'I23-\d{4}',         # I23-xxxx format (uppercase variant)
        ]
    
    def find_artwork_references(self, text_column: str = 'message_text') -> pd.DataFrame:
        """
        Find all emails that reference artwork IDs
        
        Returns:
            DataFrame with emails containing artwork references
        """
        # Combine all patterns into one regex
        combined_pattern = '|'.join(self.artwork_id_patterns)
        
        # Create a function to extract IDs from text
        def extract_artwork_ids(text):
            if pd.isna(text):
                return []
            matches = re.findall(combined_pattern, str(text), re.IGNORECASE)
            return list(set(matches))  # Remove duplicates
        
        # Apply extraction to all emails
        self.df['artwork_ids'] = self.df[text_column].apply(extract_artwork_ids)
        
        # Filter to emails with artwork references
        artwork_emails = self.df[self.df['artwork_ids'].apply(len) > 0].copy()
        
        return artwork_emails
    
    def search_by_artwork_id(self, artwork_id: str, text_column: str = 'message_text') -> pd.DataFrame:
        """Search for emails mentioning a specific artwork ID"""
        pattern = re.escape(artwork_id)
        mask = self.df[text_column].str.contains(pattern, case=False, na=False)
        return self.df[mask]
    
    def boolean_search(self, 
                      terms: List[str], 
                      operator: str = 'AND',
                      text_column: str = 'message_text') -> pd.DataFrame:
        """
        Perform boolean search across emails
        
        Args:
            terms: List of search terms
            operator: 'AND', 'OR', or 'NOT'
            text_column: Column to search in
        """
        if not terms:
            return pd.DataFrame()
        
        masks = []
        for term in terms:
            mask = self.df[text_column].str.contains(term, case=False, na=False)
            masks.append(mask)
        
        if operator.upper() == 'AND':
            final_mask = masks[0]
            for mask in masks[1:]:
                final_mask = final_mask & mask
        elif operator.upper() == 'OR':
            final_mask = masks[0]
            for mask in masks[1:]:
                final_mask = final_mask | mask
        elif operator.upper() == 'NOT':
            # NOT operation: first term must be present, others must not
            final_mask = masks[0]
            for mask in masks[1:]:
                final_mask = final_mask & (~mask)
        else:
            raise ValueError("Operator must be 'AND', 'OR', or 'NOT'")
        
        return self.df[final_mask]
    
    def regex_search(self, pattern: str, text_column: str = 'message_text') -> pd.DataFrame:
        """Perform regex search across emails"""
        try:
            mask = self.df[text_column].str.contains(pattern, case=False, na=False, regex=True)
            return self.df[mask]
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")
    
    def search_by_date_range(self, 
                           start_date: Optional[str] = None, 
                           end_date: Optional[str] = None,
                           date_column: str = 'date') -> pd.DataFrame:
        """Filter emails by date range"""
        df_filtered = self.df.copy()
        
        # Convert date column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df_filtered[date_column]):
            df_filtered[date_column] = pd.to_datetime(df_filtered[date_column], errors='coerce')
        
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df_filtered = df_filtered[df_filtered[date_column] >= start_dt]
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            df_filtered = df_filtered[df_filtered[date_column] <= end_dt]
        
        return df_filtered
    
    def get_artwork_mention_summary(self, text_column: str = 'message_text') -> Dict[str, Any]:
        """Get summary statistics of artwork mentions"""
        artwork_emails = self.find_artwork_references(text_column)
        
        # Flatten all artwork IDs
        all_artwork_ids = []
        for ids_list in artwork_emails['artwork_ids']:
            all_artwork_ids.extend(ids_list)
        
        # Count occurrences
        from collections import Counter
        id_counts = Counter(all_artwork_ids)
        
        return {
            'total_emails_with_artwork_refs': len(artwork_emails),
            'unique_artwork_ids_mentioned': len(id_counts),
            'most_mentioned_artworks': id_counts.most_common(10),
            'artwork_id_breakdown': dict(id_counts)
        }



# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """Example of how to use the modular system"""
    
    # Initialize individual modules
    print("=== Testing Email Text Processor ===")
    email_processor = EmailTextProcessor("./email_related/machnik_emails.csv")
    
    # Find artwork references
    artwork_refs = email_processor.find_artwork_references()
    print(f"Found {len(artwork_refs)} emails with artwork references")
    
    # Get summary
    summary = email_processor.get_artwork_mention_summary()
    print(f"Summary: {summary}")


if __name__ == "__main__":
    example_usage()

# #%%
# email_processor = EmailTextProcessor("./machnik_emails.csv")
# df = email_processor.search_by_date_range(start_date='2025-03-26',end_date='2025-06-18')
# df = email_processor.boolean_search(['Mila','Tom','Paul','George'])
