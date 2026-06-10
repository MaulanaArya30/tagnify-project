#Main Schema classes for Tagnify

from __future__ import annotations
from pydantic import BaseModel, field_validator, model_validator, ConfigDict


# Example
class Example(BaseModel):
    """Mandatory single few-shot example to teach the model to label
    
    Args:
        text: The input text being labeled.
        label: The correct label for the input text.
        reasoning: Optional explanation for why the label is correct.

    Example:
        Example(
            text="The movie was fantastic and I loved it!",
            label="positive",
            reasoning="The text expresses a positive sentiment towards the movie."
        )
    """
    model_config = ConfigDict(frozen=True) #immutable dataclass

    text: str
    label: str
    reasoning: str | None = None

# Schema
class Schema(BaseModel):
    """Defines the schema for labeling tasks, including the label set and mandatory few-shot examples.
    
    Args:
        labels: A list of valid labels for the classification task.
        examples: A list of few-shot examples to guide the model in labeling.
        description: Optional description of the labeling task.
        confidence_threshold: Result below this confidence level will be flagged for review, default is 0.5.

    Example:
        Schema(
            labels=["positive", "negative", "neutral"],
            examples=[
                Example(
                    text="The movie was fantastic and I loved it!",
                    label="positive",
                    reasoning="The text expresses a positive sentiment towards the movie."
                ),
                Example(
                    text="The movie was terrible and I hated it.",
                    label="negative",
                    reasoning="The text expresses a negative sentiment towards the movie."
                )
            ],
            description="Classify movie reviews as positive, negative, or neutral.",
            confidence_threshold=0.7
        )
    """
    labels: list[str]
    examples: list[Example]
    description: str | None = None
    confidence_threshold: float = 0.5


    # -- Field Validators --

    @field_validator('labels')
    @classmethod
    def labels_must_be_valid(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("Labels list cannot be empty.")
        if len(v) < 2:
            raise ValueError(f"At least two labels are required for classification. Only {len(v)} provided.")
        if len(set(v)) != len(v):
            seen = set()
            duplicates = set(x for x in v if x in seen or seen.add(x))
            raise ValueError(f"Labels must be unique. Duplicates found: {', '.join(duplicates)}")
        return v

    @field_validator('confidence_threshold')
    @classmethod
    def threshold_must_be_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence threshold must be a float between 0.0 and 1.0.")
        return v    

    @field_validator('examples')
    @classmethod
    def examples_must_not_be_empty(cls, v: list[Example]) -> list[Example]:
        if not v:
            raise ValueError("Examples list is Mandatory and cannot be empty.")
        return v


    # -- Cross-field Validators (after fields are set) --

    @model_validator(mode='after')
    def example_labels_must_exist_in_labels(self) -> Schema:
        """Cross-validate that every example's label is included in the defined labels list."""
        invalid = [
            ex.label for ex in self.examples 
            if ex.label not in self.labels
        ]
        if invalid:
            raise ValueError(f"Example labels must be in the defined labels. Invalid labels found: {', '.join(set(invalid))}")
        return self


    # -- Dict escape hatch --

    @classmethod
    def from_dict(cls, data: dict) -> Schema:
        """Create a Schema instance from a dictionary, allowing for flexible input formats.
        
        Example:
                >>> schema = Schema.from_dict({
                ...     "labels": ["positive", "negative"],
                ...     "examples": [
                ...         {"text": "Great!", "label": "positive"}
                ...     ]
                ... })
        """
        examples_raw = data.get("examples", [])
        examples = [
            Example(**ex) if isinstance(ex, dict) else ex 
            for ex in examples_raw
        ]
        return cls.model_validate({**data, "examples": examples})


# Label Result
class LabelResult(BaseModel):
    """Represents the output of a single labeling task.
    
    Fields:
        label: The label assigned by the model. None when success=False.
        confidence: The confidence score for the assigned label, between 0 and 1.
        reasoning: Optional explanation for why the label was assigned.
        flagged: True when confidence is below the threshold, indicating the result should be flagged for review.
        attempts: Number of attempts taken to get a valid label (for internal use).
        success: False when all retries have been exhausted without getting a valid label, True otherwise.
        error: Readable error message when success=False, None otherwise.

    Example:
        LabelResult(
            label="positive",
            confidence=0.85,
            reasoning="The text expresses a positive sentiment towards the movie.",
            flagged=False,
            attempts=1,
            success=True,
            error=None
        )
    """
    model_config = ConfigDict(frozen=True) #immutable dataclass

    label: str | None
    confidence: float
    reasoning: str | None = None
    flagged: bool = False
    attempts: int = 1
    success: bool = True
    error: str | None = None