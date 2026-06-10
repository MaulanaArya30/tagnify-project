"""Tests for tagnify.schema"""

import pytest
from tagnify.schema import Example, Schema, LabelResult


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def make_example(**kwargs) -> Example:
    """Minimal valid Example with overridable fields."""
    defaults = {"text": "Great product!", "label": "positive"}
    return Example(**{**defaults, **kwargs})


def make_schema(**kwargs) -> Schema:
    """Minimal valid Schema with overridable fields."""
    defaults = {
        "labels": ["positive", "negative"],
        "examples": [make_example()],
    }
    return Schema(**{**defaults, **kwargs})


# ═══════════════════════════════════════════════════════════════
# Example
# ═══════════════════════════════════════════════════════════════

class TestExample:

    def test_minimal_creation(self):
        ex = Example(text="Great product!", label="positive")
        assert ex.text == "Great product!"
        assert ex.label == "positive"
        assert ex.reasoning is None  # optional, defaults to None

    def test_with_reasoning(self):
        ex = Example(
            text="Terrible experience.",
            label="negative",
            reasoning="The word 'terrible' is a strong negative signal."
        )
        assert ex.reasoning == "The word 'terrible' is a strong negative signal."

    def test_is_immutable(self):
        """Example should be frozen — results shouldn't be mutated after creation."""
        ex = make_example()
        with pytest.raises(Exception):  # pydantic raises ValidationError on frozen models
            ex.text = "changed"

    def test_is_immutable_label(self):
        ex = make_example()
        with pytest.raises(Exception):
            ex.label = "negative"


# ═══════════════════════════════════════════════════════════════
# Schema — valid cases
# ═══════════════════════════════════════════════════════════════

class TestSchemaValid:

    def test_minimal_valid_schema(self):
        schema = make_schema()
        assert schema.labels == ["positive", "negative"]
        assert len(schema.examples) == 1
        assert schema.confidence_threshold == 0.5  # default
        assert schema.description is None           # optional

    def test_multiple_labels(self):
        schema = make_schema(labels=["positive", "negative", "neutral"])
        assert len(schema.labels) == 3

    def test_multiple_examples(self):
        schema = make_schema(
            examples=[
                Example(text="Great!", label="positive"),
                Example(text="Terrible!", label="negative"),
            ]
        )
        assert len(schema.examples) == 2

    def test_custom_confidence_threshold(self):
        schema = make_schema(confidence_threshold=0.8)
        assert schema.confidence_threshold == 0.8

    def test_threshold_at_zero(self):
        schema = make_schema(confidence_threshold=0.0)
        assert schema.confidence_threshold == 0.0

    def test_threshold_at_one(self):
        schema = make_schema(confidence_threshold=1.0)
        assert schema.confidence_threshold == 1.0

    def test_with_description(self):
        schema = make_schema(description="Customer reviews for a food delivery app")
        assert schema.description == "Customer reviews for a food delivery app"

    def test_example_with_reasoning_allowed(self):
        """Examples with reasoning are valid."""
        schema = make_schema(
            examples=[
                Example(
                    text="Amazing!",
                    label="positive",
                    reasoning="Exclamation with positive adjective."
                )
            ]
        )
        assert schema.examples[0].reasoning is not None


# ═══════════════════════════════════════════════════════════════
# Schema — invalid cases (field validators)
# ═══════════════════════════════════════════════════════════════

class TestSchemaInvalidLabels:

    def test_rejects_single_label(self):
        """A schema with one label can't classify anything."""
        with pytest.raises(ValueError, match="At least two labels"):
            make_schema(labels=["positive"])

    def test_rejects_empty_labels(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            make_schema(labels=[])

    def test_rejects_duplicate_labels(self):
        with pytest.raises(ValueError, match="unique"):
            make_schema(labels=["positive", "positive", "negative"])

    def test_rejects_all_duplicate_labels(self):
        with pytest.raises(ValueError, match="unique"):
            make_schema(labels=["positive", "positive"])


class TestSchemaInvalidExamples:

    def test_rejects_empty_examples(self):
        """Core design stance: examples are non-negotiable."""
        with pytest.raises(ValueError, match="Mandatory"):
            make_schema(examples=[])


class TestSchemaInvalidThreshold:

    def test_rejects_threshold_above_one(self):
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            make_schema(confidence_threshold=1.5)

    def test_rejects_threshold_below_zero(self):
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            make_schema(confidence_threshold=-0.1)


# ═══════════════════════════════════════════════════════════════
# Schema — cross-field validation (model_validator)
# ═══════════════════════════════════════════════════════════════

class TestSchemaCrossValidation:

    def test_rejects_example_label_not_in_labels(self):
        with pytest.raises(ValueError, match="must be in the defined labels"):
            Schema(
                labels=["positive", "negative"],
                examples=[Example(text="Meh.", label="neutral")]  # "neutral" not in labels
            )

    def test_rejects_multiple_invalid_example_labels(self):
        with pytest.raises(ValueError, match="must be in the defined labels"):
            Schema(
                labels=["positive", "negative"],
                examples=[
                    Example(text="OK", label="neutral"),     # invalid
                    Example(text="Fine", label="mixed"),     # invalid
                ]
            )

    def test_accepts_example_labels_that_exist(self):
        """All example labels are in the labels list — should pass."""
        schema = Schema(
            labels=["positive", "negative", "neutral"],
            examples=[
                Example(text="Great!", label="positive"),
                Example(text="Terrible!", label="negative"),
                Example(text="Fine.", label="neutral"),
            ]
        )
        assert len(schema.examples) == 3


# ═══════════════════════════════════════════════════════════════
# Schema.from_dict — dict escape hatch
# ═══════════════════════════════════════════════════════════════

class TestSchemaFromDict:

    def test_basic_from_dict(self):
        schema = Schema.from_dict({
            "labels": ["positive", "negative"],
            "examples": [{"text": "Great!", "label": "positive"}],
        })
        assert isinstance(schema, Schema)
        assert isinstance(schema.examples[0], Example)
        assert schema.examples[0].text == "Great!"

    def test_from_dict_with_all_fields(self):
        schema = Schema.from_dict({
            "labels": ["positive", "negative", "neutral"],
            "description": "Product reviews",
            "confidence_threshold": 0.75,
            "examples": [
                {"text": "Great!", "label": "positive"},
                {"text": "Terrible!", "label": "negative", "reasoning": "Strong negative."},
            ],
        })
        assert schema.confidence_threshold == 0.75
        assert schema.description == "Product reviews"
        assert schema.examples[1].reasoning == "Strong negative."

    def test_from_dict_runs_same_validation(self):
        """from_dict is syntax sugar, not a validation bypass."""
        with pytest.raises(ValueError, match="Mandatory"):
            Schema.from_dict({
                "labels": ["positive", "negative"],
                "examples": [],
            })

    def test_from_dict_cross_validates_example_labels(self):
        with pytest.raises(ValueError, match="must be in the defined labels"):
            Schema.from_dict({
                "labels": ["positive", "negative"],
                "examples": [{"text": "Meh.", "label": "neutral"}],
            })

    def test_from_dict_accepts_already_instantiated_examples(self):
        """If examples are already Example objects, from_dict handles that too."""
        schema = Schema.from_dict({
            "labels": ["positive", "negative"],
            "examples": [Example(text="Great!", label="positive")],
        })
        assert isinstance(schema.examples[0], Example)


# ═══════════════════════════════════════════════════════════════
# LabelResult
# ═══════════════════════════════════════════════════════════════

class TestLabelResult:

    def test_successful_result(self):
        result = LabelResult(label="positive", confidence=0.92)
        assert result.label == "positive"
        assert result.confidence == 0.92
        assert result.success is True       # default
        assert result.flagged is False      # default
        assert result.attempts == 1         # default
        assert result.reasoning is None     # default
        assert result.error is None         # default

    def test_failed_result(self):
        """Failure: all retries exhausted. Return a result, never raise."""
        result = LabelResult(
            label=None,
            confidence=0.0,
            success=False,
            attempts=3,
            error="LLM returned 'maybe' — not in labels list ['positive', 'negative']"
        )
        assert result.label is None
        assert result.success is False
        assert result.attempts == 3
        assert "maybe" in result.error

    def test_flagged_result(self):
        """Low confidence result — flagged for human review."""
        result = LabelResult(label="neutral", confidence=0.3, flagged=True)
        assert result.flagged is True
        assert result.label == "neutral"  # label is still present

    def test_result_with_reasoning(self):
        result = LabelResult(
            label="negative",
            confidence=0.87,
            reasoning="The phrase 'arrived cold' is a clear complaint."
        )
        assert result.reasoning is not None

    def test_multi_attempt_result(self):
        """A result that needed retries. Still valid — just took longer."""
        result = LabelResult(label="positive", confidence=0.78, attempts=2)
        assert result.attempts == 2
        assert result.success is True  # still succeeded, just on attempt 2

    def test_is_immutable(self):
        """Results are read-only — nobody should mutate a label result."""
        result = LabelResult(label="positive", confidence=0.9)
        with pytest.raises(Exception):
            result.label = "negative"

    def test_is_immutable_confidence(self):
        result = LabelResult(label="positive", confidence=0.9)
        with pytest.raises(Exception):
            result.confidence = 0.5