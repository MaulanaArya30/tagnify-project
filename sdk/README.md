# Tagnify

LLM-powered automated data labeling — schema-first, confidence-scored, local-model ready.

```python
from tagnify import Tagnify, Schema, Example

schema = Schema(
    labels=["positive", "negative", "neutral"],
    examples=[Example(text="Great product!", label="positive")]
)

tagnify = Tagnify(model="qwen2.5:7b")
result = tagnify.label("This was a disappointing experience.", schema)

print(result.label)       # "negative"
print(result.confidence)  # 0.91
```

## Installation

```bash
pip install tagnify
```

Requires [Ollama](https://ollama.ai) running locally.