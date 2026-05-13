"""Shared test fixtures for the Convolve MAS backend."""

import pytest


@pytest.fixture
def sample_text_neutral():
    """A neutral text message for testing."""
    return "I went to the store today and picked up some groceries."


@pytest.fixture
def sample_text_crisis():
    """A crisis-level text message for testing."""
    return "I want to kill myself. I can't take this anymore."


@pytest.fixture
def sample_text_moderate():
    """A moderate distress text message for testing."""
    return "I feel completely alone and nothing matters anymore."


@pytest.fixture
def sample_text_positive():
    """A positive text message for testing."""
    return "I'm feeling so much better today, things are looking great!"


@pytest.fixture
def sample_text_masking():
    """A text that sounds positive but could indicate masking."""
    return "I'm fine, everything is totally fine, don't worry about me."
