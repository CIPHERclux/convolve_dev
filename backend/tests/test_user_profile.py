"""
Tests for UserProfile — deterministic fact extraction.

Tests name, age, location, and diagnosis extraction from natural
language input. All regex-based, no external dependencies.
"""

from unittest.mock import patch

import pytest

from app.memory.user_profile import UserProfile


@pytest.fixture
def profile(tmp_path):
    """Create a UserProfile with a temporary storage directory."""
    with patch.object(UserProfile, "__init__", lambda self, uid: None):
        p = UserProfile.__new__(UserProfile)

    p.user_id = "test_user"
    p.path = str(tmp_path / "test_profile.json")
    p.facts = {
        "name": None,
        "pronouns": None,
        "age": None,
        "location": None,
        "occupation": None,
        "school": None,
        "relationships": {},
        "diagnoses": [],
        "medications": [],
        "goals": [],
        "main_concerns": [],
        "custom_facts": {},
        "first_interaction": None,
        "last_updated": None,
        "total_interactions": 0,
    }
    p._name_exclusions = {
        "i",
        "me",
        "my",
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "so",
        "just",
        "really",
        "actually",
        "basically",
        "well",
        "yeah",
        "yes",
        "no",
        "not",
        "hey",
        "hi",
        "hello",
        "bye",
        "good",
        "bad",
        "fine",
        "okay",
        "ok",
        "happy",
        "sad",
        "angry",
        "scared",
        "worried",
        "anxious",
        "depressed",
        "tired",
        "confused",
        "alone",
        "lonely",
        "morning",
        "evening",
        "night",
    }
    return p


# ─── Name Extraction ────────────────────────────────────────────────────


class TestNameExtraction:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("My name is Sarah", "Sarah"),
            ("I'm Alex", "Alex"),
            ("Call me Jordan", "Jordan"),
        ],
    )
    def test_name_patterns(self, profile, text, expected):
        facts = profile.extract_facts(text)
        assert profile.facts["name"] == expected
        assert any(f["type"] == "name" for f in facts)

    def test_name_exclusion(self, profile):
        """Common words should not be detected as names."""
        profile.extract_facts("Hello")
        assert profile.facts["name"] is None

    def test_name_only_extracted_once(self, profile):
        """Name should not be overwritten once set."""
        profile.extract_facts("My name is Sarah")
        profile.extract_facts("My name is Emily")
        assert profile.facts["name"] == "Sarah"

    def test_name_requires_uppercase(self, profile):
        profile.extract_facts("my name is sarah")
        # Pattern requires capitalized first letter
        assert profile.facts["name"] is None or profile.facts["name"][0].isupper()


# ─── Age Extraction ──────────────────────────────────────────────────────


class TestAgeExtraction:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("I'm 25 years old", 25),
            ("I am 30 years old", 30),
            ("im 18 yo", 18),
        ],
    )
    def test_age_patterns(self, profile, text, expected):
        profile.extract_facts(text)
        assert profile.facts["age"] == expected

    def test_invalid_age_rejected(self, profile):
        """Ages outside 10-100 should be rejected."""
        profile.extract_facts("I'm 5 years old")
        assert profile.facts["age"] is None

    def test_age_only_extracted_once(self, profile):
        profile.extract_facts("I'm 25 years old")
        profile.extract_facts("I'm 30 years old")
        assert profile.facts["age"] == 25


# ─── Location Extraction ────────────────────────────────────────────────


class TestLocationExtraction:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("i live in New York.", "New York"),
            ("i'm from San Francisco.", "San Francisco"),
        ],
    )
    def test_location_patterns(self, profile, text, expected):
        profile.extract_facts(text)
        assert profile.facts["location"] == expected


# ─── Diagnosis Extraction ───────────────────────────────────────────────


class TestDiagnosisExtraction:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("I have depression", "depression"),
            ("I was diagnosed with anxiety", "anxiety"),
            ("I've been diagnosed with PTSD", "ptsd"),
            ("I struggle with OCD", "ocd"),
        ],
    )
    def test_diagnosis_patterns(self, profile, text, expected):
        profile.extract_facts(text)
        assert expected in profile.facts["diagnoses"]

    def test_no_duplicate_diagnoses(self, profile):
        profile.extract_facts("I have depression")
        profile.extract_facts("I have depression")
        assert profile.facts["diagnoses"].count("depression") == 1

    def test_multiple_diagnoses(self, profile):
        profile.extract_facts("I have depression")
        profile.extract_facts("I was diagnosed with anxiety")
        assert "depression" in profile.facts["diagnoses"]
        assert "anxiety" in profile.facts["diagnoses"]


# ─── Profile Summary ────────────────────────────────────────────────────


class TestProfileSummary:
    def test_summary_empty_profile(self, profile):
        summary = profile.get_summary_for_llm()
        assert summary == ""

    def test_summary_with_facts(self, profile):
        profile.facts["name"] = "Sarah"
        profile.facts["age"] = 25
        summary = profile.get_summary_for_llm()
        assert "Sarah" in summary
        assert "25" in summary
        assert "USER PROFILE" in summary


# ─── Edge Cases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_text(self, profile):
        facts = profile.extract_facts("")
        assert facts == []

    def test_whitespace_text(self, profile):
        facts = profile.extract_facts(" ")
        assert facts == []

    def test_interaction_counter(self, profile):
        profile.extract_facts("Hello there, nice to meet you")
        profile.extract_facts("I had a good day")
        assert profile.facts["total_interactions"] == 2

    def test_set_custom_fact(self, profile):
        profile.set_fact("hobby", "painting")
        assert profile.facts["custom_facts"]["hobby"] == "painting"

    def test_set_known_fact(self, profile):
        profile.set_fact("occupation", "engineer")
        assert profile.facts["occupation"] == "engineer"
