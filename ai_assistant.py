#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 14:16:21 2025

@author: prcohen
"""

#!/usr/bin/env python3
"""
AI Assistant for Harold Cohen Research
Clean integration with Anthropic's Claude API
"""

import streamlit as st
from typing import List, Dict, Any, Optional
from datetime import datetime

# Try to import anthropic
try:
    import anthropic
    anthropic_available = True
except ImportError:
    anthropic_available = False


class HaroldCohenAI:
    """AI assistant specialized for Harold Cohen research"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = None
        self.api_key = api_key
        self.cost_estimate = 0.0
        self._initialize()
    
    def _initialize(self):
        """Initialize the Anthropic client"""
        try:
            if not anthropic_available:
                return False
            
            # Get API key from multiple sources
            if not self.api_key:
                # Try Streamlit secrets first
                self.api_key = st.secrets.get("ANTHROPIC_API_KEY")
            
            if not self.api_key:
                return False
            
            # Initialize client
            self.client = anthropic.Anthropic(api_key=self.api_key)
            return True
            
        except Exception as e:
            return False
    
    def is_ready(self) -> bool:
        """Check if AI assistant is ready to use"""
        return self.client is not None
    
    def analyze_search_results(self, query: str, search_results: List[Dict], max_results: int = 3) -> Dict[str, Any]:
        """Analyze search results and provide intelligent insights"""
        try:
            if not self.is_ready():
                return {"status": "error", "message": "AI assistant not available"}
            
            if not search_results:
                return {"status": "error", "message": "No search results to analyze"}
            
            # Prepare context from search results
            context_pieces = []
            for i, result in enumerate(search_results[:max_results]):
                content = result.get('content', '')[:500]  # Limit length
                source = result.get('metadata', {}).get('source', 'Unknown')
                context_pieces.append(f"Document {i+1} (from {source}):\n{content}")
            
            context = "\n\n".join(context_pieces)
            
            # Create the prompt
            prompt = f"""You are an expert research assistant specializing in Harold Cohen's work and life. Harold Cohen (1928-2016) was a pioneering computer artist who created the AARON drawing program.

SEARCH QUERY: {query}

RELEVANT DOCUMENTS:
{context}

Please provide a comprehensive analysis that:
1. Directly answers the user's query based on the documents
2. Synthesizes information across the documents when relevant
3. Highlights key insights about Harold Cohen's work, exhibitions, or artistic process
4. Notes any important dates, locations, people, or artworks mentioned
5. Suggests related questions the researcher might want to explore

Keep your response scholarly but accessible, and focus specifically on what the documents reveal about Harold Cohen's artistic practice and career."""

            # Make API call
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",  # Use faster, cheaper model for initial implementation
                max_tokens=1000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract response text
            response_text = response.content[0].text
            
            # Rough cost estimate (Haiku pricing: ~$0.25/$1.25 per 1M tokens)
            input_tokens = len(prompt.split()) * 1.3  # Rough estimate
            output_tokens = len(response_text.split()) * 1.3
            estimated_cost = (input_tokens * 0.00000025) + (output_tokens * 0.00000125)
            self.cost_estimate += estimated_cost
            
            return {
                "status": "success",
                "response": response_text,
                "cost_estimate": estimated_cost,
                "total_cost": self.cost_estimate,
                "model": "claude-3-haiku-20240307",
                "query": query,
                "documents_analyzed": len(search_results[:max_results])
            }
            
        except Exception as e:
            return {"status": "error", "message": f"AI analysis failed: {str(e)}"}
    
    def suggest_follow_up_queries(self, original_query: str, ai_response: str) -> List[str]:
        """Suggest follow-up research questions"""
        try:
            if not self.is_ready():
                return []
            
            prompt = f"""Based on this Harold Cohen research query and response, suggest 3-5 specific follow-up questions that would help deepen the research.

ORIGINAL QUERY: {original_query}

AI RESPONSE: {ai_response}

Suggest follow-up questions that are:
- Specific to Harold Cohen's work and life
- Based on themes or details mentioned in the response
- Likely to yield interesting results in an email/document archive
- Formatted as direct questions

Return only the questions, one per line, without numbering."""

            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=300,
                temperature=0.4,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse suggestions
            suggestions = response.content[0].text.strip().split('\n')
            return [q.strip() for q in suggestions if q.strip()]
            
        except Exception as e:
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get AI assistant statistics"""
        return {
            "available": self.is_ready(),
            "total_cost_estimate": self.cost_estimate,
            "model": "claude-3-haiku-20240307" if self.is_ready() else None,
            "library_available": anthropic_available
        }


# Streamlit cache for the AI assistant
@st.cache_resource
def get_ai_assistant():
    """Get cached AI assistant instance"""
    return HaroldCohenAI()
