"""
Tests for tagnify.prompt — PromptBuilder.

PromptBuilder is pure string manipulation — no HTTP, no models,
no external dependencies. Every test just builds a prompt and
asserts on the resulting string.
"""

import pytest
from tagnify.prompt import PromptBuilder
from tagnify.schema import Schema, Example


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def make_schema(**kwargs) -> Schema:
    defaults = {
        "labels": ["positive", "negative"],
        "examples": [Example(text="Great product!", label="positive")],
    }
    return Schema(**{**defaults, **kwargs})


# ═══════════════════════════════════════════════════════════════
# Core content
# ═══════════════════════════════════════════════════════════════

class TestPromptBuilderContent:

    def test_prompt_contains_all_labels(self):
        schema = make_schema(labels=["positive", "negative", "neutral"])
        prompt = PromptBuilder().build("test", schema)
        assert "positive" in prompt
        assert "negative" in prompt
        assert "neutral" in prompt

    def test_prompt_contains_example_text(self):
        schema = make_schema(
            examples=[Example(text="Arrived hot and on time!", label="positive")]
        )
        prompt = PromptBuilder().build("test", schema)
        assert "Arrived hot and on time!" in prompt

    def test_prompt_contains_example_label(self):
        schema = make_schema(
            examples=[Example(text="Terrible service.", label="negative")]
        )
        prompt = PromptBuilder().build("test", schema)
        assert "negative" in prompt

    def test_prompt_contains_input_text(self):
        prompt = PromptBuilder().build("My unique input text", make_schema())
        assert "My unique input text" in prompt

    def test_prompt_contains_json_instruction(self):
        prompt = PromptBuilder().build("test", make_schema())
        assert "JSON" in prompt
        assert "confidence" in prompt

    def test_prompt_contains_format_rules(self):
        """The prompt must explicitly forbid markdown fences."""
        prompt = PromptBuilder().build("test", make_schema())
        assert "markdown" in prompt.lower()


# ═══════════════════════════════════════════════════════════════
# Description field
# ═══════════════════════════════════════════════════════════════

class TestPromptBuilderDescription:

    def test_description_included_when_provided(self):
        schema = make_schema(description="Food delivery app reviews")
        prompt = PromptBuilder().build("test", schema)
        assert "Food delivery app reviews" in prompt

    def test_description_absent_when_none(self):
        schema = make_schema(description=None)
        prompt = PromptBuilder().build("test", schema)
        assert "Task context" not in prompt


# ═══════════════════════════════════════════════════════════════
# Reasoning flag
# ═══════════════════════════════════════════════════════════════

class TestPromptBuilderReasoning:

    def test_reasoning_field_absent_by_default(self):
        prompt = PromptBuilder().build("test", make_schema(), reasoning=False)
        # The JSON template should not include a reasoning key
        assert '"reasoning"' not in prompt

    def test_reasoning_field_present_when_enabled(self):
        prompt = PromptBuilder().build("test", make_schema(), reasoning=True)
        assert '"reasoning"' in prompt

    def test_example_reasoning_injected_when_present(self):
        schema = make_schema(
            examples=[
                Example(
                    text="Perfect delivery.",
                    label="positive",
                    reasoning="Explicit praise with no qualifiers.",
                )
            ]
        )
        prompt = PromptBuilder().build("test", schema, reasoning=True)
        assert "Explicit praise with no qualifiers." in prompt

    def test_example_reasoning_not_injected_when_reasoning_false(self):
        schema = make_schema(
            examples=[
                Example(
                    text="Perfect delivery.",
                    label="positive",
                    reasoning="This reasoning should not appear.",
                )
            ]
        )
        prompt = PromptBuilder().build("test", schema, reasoning=False)
        assert "This reasoning should not appear." not in prompt


# ═══════════════════════════════════════════════════════════════
# Retry reminders
# ═══════════════════════════════════════════════════════════════

class TestPromptBuilderRetry:

    def test_no_reminder_on_attempt_1(self):
        prompt = PromptBuilder().build("test", make_schema(), attempt=1)
        assert "IMPORTANT" not in prompt
        assert "CRITICAL" not in prompt

    def test_important_reminder_on_attempt_2(self):
        prompt = PromptBuilder().build("test", make_schema(), attempt=2)
        assert "IMPORTANT" in prompt

    def test_critical_reminder_on_attempt_3(self):
        prompt = PromptBuilder().build("test", make_schema(), attempt=3)
        assert "CRITICAL" in prompt

    def test_attempt_3_reminder_stronger_than_attempt_2(self):
        prompt_2 = PromptBuilder().build("test", make_schema(), attempt=2)
        prompt_3 = PromptBuilder().build("test", make_schema(), attempt=3)
        assert "IMPORTANT" in prompt_2
        assert "CRITICAL" in prompt_3
        assert "IMPORTANT" not in prompt_3  # escalated, not duplicated


# ═══════════════════════════════════════════════════════════════
# Multiple examples
# ═══════════════════════════════════════════════════════════════

class TestPromptBuilderMultipleExamples:

    def test_all_examples_included(self):
        schema = make_schema(
            examples=[
                Example(text="Loved it!", label="positive"),
                Example(text="Hated it!", label="negative"),
                Example(text="It was fine.", label="neutral"),
            ],
            labels=["positive", "negative", "neutral"],
        )
        prompt = PromptBuilder().build("test", schema)
        assert "Loved it!" in prompt
        assert "Hated it!" in prompt
        assert "It was fine." in prompt

    def test_prompt_is_a_string(self):
        prompt = PromptBuilder().build("test", make_schema())
        assert isinstance(prompt, str)
        assert len(prompt) > 0