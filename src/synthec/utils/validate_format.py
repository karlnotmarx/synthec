# utils/validate_schema.py
from jsonschema import validate, ValidationError

# Define the exact JSON structure expected
SCHEMA = {
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "paragraph": {"type": "string"},
      "label": {"type": "string", "enum": ["positive", "neutral", "negative"]}
    },
    "required": ["paragraph", "label"]
  }
}

def validate_json(data):
    """Validates the generated JSON array against the predefined SCHEMA."""
    try:
        validate(instance=data, schema=SCHEMA)
        return True, None
    except ValidationError as e:
        return False, str(e)