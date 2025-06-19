#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 20:40:01 2025

@author: prcohen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 20:12:37 2025

@author: prcohen
"""

#!/usr/bin/env python3
"""
Modular Catalogue Raisonné System for Harold Cohen
Artwork metadata module
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



# ============================================================================
# 3. ARTWORK METADATA MODULE
# ============================================================================

@dataclass
class ArtworkRecord:
    """Artwork metadata structure"""
    id: str
    title: str
    year: Optional[int]
    date_range: Optional[str]
    medium: str
    dimensions: Optional[str]
    description: Optional[str]
    image_path: Optional[str]
    provenance: Optional[str]
    exhibitions: List[str]
    notes: Optional[str]

class ArtworkDatabase:
    """
    Module for managing artwork metadata
    """
    
    def __init__(self):
        self.artworks: Dict[str, ArtworkRecord] = {}
    
    def load_from_csv(self, csv_path: str) -> int:
        """Load artwork metadata from CSV file"""
        df = pd.read_csv(csv_path)
        loaded_count = 0
        
        for _, row in df.iterrows():
            artwork = ArtworkRecord(
                id=str(row['id']),
                title=row.get('title', 'Untitled'),
                year=int(row['year']) if pd.notna(row.get('year')) else None,
                date_range=row.get('date_range'),
                medium=row.get('medium', ''),
                dimensions=row.get('dimensions'),
                description=row.get('description'),
                image_path=row.get('image_path'),
                provenance=row.get('provenance'),
                exhibitions=row.get('exhibitions', '').split(';') if row.get('exhibitions') else [],
                notes=row.get('notes')
            )
            self.artworks[artwork.id] = artwork
            loaded_count += 1
        
        return loaded_count
    
    def get_artwork(self, artwork_id: str) -> Optional[ArtworkRecord]:
        """Get artwork by ID"""
        return self.artworks.get(artwork_id)
    
    def search_by_title(self, title_query: str) -> List[ArtworkRecord]:
        """Search artworks by title"""
        results = []
        title_lower = title_query.lower()
        
        for artwork in self.artworks.values():
            if title_lower in artwork.title.lower():
                results.append(artwork)
        
        return results
    
    def get_artworks_by_year_range(self, start_year: int, end_year: int) -> List[ArtworkRecord]:
        """Get artworks within a year range"""
        results = []
        
        for artwork in self.artworks.values():
            if artwork.year and start_year <= artwork.year <= end_year:
                results.append(artwork)
        
        return results



# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """Example of how to use the modular system"""
    
    pass

if __name__ == "__main__":
    example_usage()
