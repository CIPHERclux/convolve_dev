"""
User Profile — deterministic fact storage for exact retrieval.

Simplified from original: same extraction logic, cleaner structure.
"""

import json
import re
import os
from typing import Dict, List, Optional, Any
from datetime import datetime

from app.config import settings
from app.utils.logger import get_logger

log = get_logger("memory.profile")


class UserProfile:
    """O(1) fact retrieval — names, relationships, diagnoses, etc."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        os.makedirs(settings.PROFILES_DIR, exist_ok=True)
        self.path = os.path.join(settings.PROFILES_DIR, f"{user_id}_profile.json")

        self.facts: Dict[str, Any] = {
            "name": None, "pronouns": None, "age": None,
            "location": None, "occupation": None, "school": None,
            "relationships": {},
            "diagnoses": [], "medications": [],
            "goals": [], "main_concerns": [],
            "custom_facts": {},
            "first_interaction": None, "last_updated": None,
            "total_interactions": 0,
        }

        self._name_exclusions = {
            "i", "me", "my", "the", "a", "an", "and", "or", "but", "so", "just",
            "really", "actually", "basically", "well", "yeah", "yes", "no", "not",
            "hey", "hi", "hello", "bye", "good", "bad", "fine", "okay", "ok",
            "happy", "sad", "angry", "scared", "worried", "anxious", "depressed",
            "tired", "confused", "alone", "lonely", "morning", "evening", "night",
        }

        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    data = json.load(f)
                for k, v in data.items():
                    if k in self.facts:
                        self.facts[k] = v
                log.info(f"Profile loaded for {self.facts.get('name', 'unknown')}")
            except Exception as e:
                log.warning(f"Could not load profile: {e}")
        else:
            self.facts["first_interaction"] = datetime.now().isoformat()

    def _save(self):
        try:
            self.facts["last_updated"] = datetime.now().isoformat()
            with open(self.path, "w") as f:
                json.dump(self.facts, f, indent=2, default=str)
        except Exception as e:
            log.warning(f"Could not save profile: {e}")

    def extract_facts(self, text: str) -> List[Dict[str, Any]]:
        """Extract facts from user text."""
        extracted = []
        if not text or len(text.strip()) < 2:
            return extracted

        self.facts["total_interactions"] += 1

        # Name
        if not self.facts["name"]:
            name = self._extract_name(text)
            if name:
                self.facts["name"] = name
                extracted.append({"type": "name", "value": name})

        # Age
        if not self.facts["age"]:
            age = self._extract_age(text)
            if age:
                self.facts["age"] = age
                extracted.append({"type": "age", "value": age})

        # Location
        if not self.facts["location"]:
            loc = self._extract_location(text)
            if loc:
                self.facts["location"] = loc
                extracted.append({"type": "location", "value": loc})

        # Diagnoses
        for d in self._extract_diagnoses(text):
            if d not in self.facts["diagnoses"]:
                self.facts["diagnoses"].append(d)
                extracted.append({"type": "diagnosis", "value": d})

        if extracted:
            self._save()
            for fact in extracted:
                log.info(f"Extracted: {fact['type']}={fact['value']}")

        return extracted

    def set_fact(self, key: str, value: Any):
        if key in self.facts:
            self.facts[key] = value
        else:
            self.facts["custom_facts"][key] = value
        self._save()

    def get_name(self) -> Optional[str]:
        return self.facts.get("name")

    def get_summary_for_llm(self) -> str:
        """Formatted facts for LLM context."""
        lines = []
        if self.facts.get("name"):
            lines.append(f"• Name: {self.facts['name']}")
        if self.facts.get("pronouns"):
            lines.append(f"• Pronouns: {self.facts['pronouns']}")
        if self.facts.get("age"):
            lines.append(f"• Age: {self.facts['age']}")
        if self.facts.get("location"):
            lines.append(f"• Location: {self.facts['location']}")
        if self.facts.get("occupation"):
            lines.append(f"• Occupation: {self.facts['occupation']}")
        if self.facts.get("relationships"):
            for r, n in self.facts["relationships"].items():
                lines.append(f"• {r}: {n}")
        if self.facts.get("diagnoses"):
            lines.append(f"• Diagnoses: {', '.join(self.facts['diagnoses'])}")
        if self.facts.get("medications"):
            lines.append(f"• Medications: {', '.join(self.facts['medications'])}")
        if not lines:
            return ""
        return "=== USER PROFILE ===\n" + "\n".join(lines)

    def get_all_facts(self) -> Dict[str, Any]:
        return {k: v for k, v in self.facts.items()
                if v and (not isinstance(v, (list, dict)) or len(v) > 0)}

    # ── Extraction helpers ───────────────────────────────────────────────

    def _extract_name(self, text: str) -> Optional[str]:
        patterns = [
            r"(?:my name is|i'm|i am|call me|they call me)\s+([A-Z][a-z]+)",
            r"^([A-Z][a-z]+)(?:\s+here)?[!.]?\s*$",
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                name = m.group(1).strip()
                if self._valid_name(name):
                    return name

        # Single capitalized word in short messages
        words = text.strip().split()
        if len(words) <= 3:
            for w in words:
                if w[0].isupper() and len(w) >= 2 and self._valid_name(w):
                    return w
        return None

    def _valid_name(self, name: str) -> bool:
        return (
            name and len(name) >= 2 and len(name) <= 20
            and name[0].isupper()
            and name.lower() not in self._name_exclusions
            and not any(c.isdigit() for c in name)
        )

    def _extract_age(self, text: str) -> Optional[int]:
        patterns = [
            r"(?:i'm|i am|im)\s+(\d{1,2})\s*(?:years?\s*old|yo|y/o)",
            r"(\d{1,2})\s*(?:years?\s*old|yo|y/o)",
            r"(?:age|aged)\s*(?:is|:)?\s*(\d{1,2})",
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                age = int(m.group(1))
                if 10 <= age <= 100:
                    return age
        return None

    def _extract_location(self, text: str) -> Optional[str]:
        patterns = [
            r"(?:i live in|i'm from|i am from|living in|based in|moved to)\s+([A-Z][a-zA-Z\s,]+?)(?:\.|,|!|\?|$)",
        ]
        for p in patterns:
            m = re.search(p, text)
            if m:
                loc = m.group(1).strip()
                if 2 < len(loc) < 50:
                    return loc
        return None

    def _extract_diagnoses(self, text: str) -> List[str]:
        pattern = r"(?:i have|diagnosed with|i've been diagnosed with|suffering from|i struggle with)\s+(depression|anxiety|bipolar|bipolar disorder|bpd|borderline|ptsd|ocd|adhd|add|social anxiety|panic disorder|insomnia)"
        return [m.lower() for m in re.findall(pattern, text, re.IGNORECASE)]
