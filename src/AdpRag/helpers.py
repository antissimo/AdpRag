import json
import re

def parse_llm_json(text: str) -> dict:
    """
    Safely parse JSON from LLM response.
    - Extracts JSON even if model adds extra text.
    - Converts "priority" to float.
    - Returns dict with 'priority' and 'reason'.
    Raises ValueError if no valid JSON found.
    """
    try:
        # Extract first {...} block from text
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON found in LLM response: {text!r}")

        json_str = match.group(0)
        data = json.loads(json_str)

        # Ensure priority is float
        priority = data.get("priority", 0.5)
        try:
            priority = float(priority)
        except Exception:
            priority = 0.5

        reason = data.get("reason", "")

        return {"priority": priority, "reason": reason}

    except Exception as e:
        raise ValueError(f"Failed to parse LLM JSON: {text!r}") from e