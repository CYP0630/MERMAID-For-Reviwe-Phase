"""
Evidence Memory module for fact-checking agent.
Provides persistent storage and keyword-based retrieval of retrieved evidence.
"""
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path


logger = logging.getLogger(__name__)


class EvidenceMemory:
    """Evidence memory with file persistence and keyword-based retrieval."""

    def __init__(self, memory_file: str = "memory/evidence_memory.json"):
        """
        Initialize the evidence memory.

        Args:
            memory_file: Path to the JSON file for persistent storage
        """
        self.memory_file = memory_file
        self.entries: List[Dict] = []
        self._load()

    def _load(self):
        """Load memory from file if it exists."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.entries = data.get("entries", [])
                logger.info(f"Loaded {len(self.entries)} entries from memory file")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load memory file: {e}. Starting with empty memory.")
                self.entries = []
        else:
            logger.info("No existing memory file found. Starting with empty memory.")
            self.entries = []

    def _save(self):
        """Persist memory to file."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)

        try:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump({"entries": self.entries}, f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved {len(self.entries)} entries to memory file")
        except IOError as e:
            logger.error(f"Failed to save memory file: {e}")

    def add(self, query: str, evidence: str, keywords: List[str]):
        """
        Add new evidence entry to memory.

        Args:
            query: The search query that produced this evidence
            evidence: The evidence text retrieved
            keywords: Keywords associated with this evidence (for retrieval)
        """
        # Normalize keywords
        normalized_keywords = [kw.lower().strip() for kw in keywords if kw.strip()]

        # Check for duplicate (same query)
        for entry in self.entries:
            if entry["query"].lower() == query.lower():
                logger.debug(f"Duplicate query found, skipping: {query[:50]}...")
                return

        entry = {
            "query": query,
            "evidence": evidence,
            "keywords": normalized_keywords,
            "timestamp": datetime.now().isoformat()
        }

        self.entries.append(entry)
        self._save()
        logger.info(f"Added new evidence to memory: query='{query[:50]}...', keywords={normalized_keywords}")

    def retrieve(self, keywords: List[str], top_k: int = 3) -> List[Dict]:
        """
        Retrieve relevant evidence by keyword matching.

        Args:
            keywords: Keywords to match against stored entries
            top_k: Maximum number of entries to return

        Returns:
            List of matching evidence entries, sorted by relevance (match count)
        """
        if not keywords or not self.entries:
            return []

        # Normalize search keywords
        search_keywords = set(kw.lower().strip() for kw in keywords if kw.strip())

        if not search_keywords:
            return []

        # Score entries by keyword overlap
        scored_entries = []
        for entry in self.entries:
            entry_keywords = set(entry.get("keywords", []))

            # Calculate match score based on keyword overlap
            # Use partial matching: check if any search keyword is contained in entry keywords
            match_score = 0
            for search_kw in search_keywords:
                for entry_kw in entry_keywords:
                    if search_kw in entry_kw or entry_kw in search_kw:
                        match_score += 1
                        break

            if match_score > 0:
                scored_entries.append((match_score, entry))

        # Sort by score (descending) and return top_k
        scored_entries.sort(key=lambda x: x[0], reverse=True)

        results = [entry for _, entry in scored_entries[:top_k]]
        logger.info(f"Retrieved {len(results)} relevant entries for keywords: {list(search_keywords)}")

        return results

    def clear(self):
        """Clear all entries from memory."""
        self.entries = []
        self._save()
        logger.info("Memory cleared")

    def size(self) -> int:
        """Return the number of entries in memory."""
        return len(self.entries)
