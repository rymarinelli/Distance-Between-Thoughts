import re

def parse_steps(text):
    """
    Parses a reasoning response into discrete steps using several heuristics.
    """
    text = text.strip()

    # Strategy 1: Numbered Steps (e.g. "1. Do this")
    pattern_seq = re.compile(r'^(?:Step\s*)?(\d+)[\.\)\:\-]\s*(.+)$', re.IGNORECASE | re.MULTILINE)
    matches = pattern_seq.findall(text)
    if matches:
        steps = []
        expected = 1
        for num, step in matches:
            if int(num) == expected:
                steps.append(step.strip())
                expected += 1
            else:
                steps = []
                break
        if steps: return steps

    # Strategy 2: Bullet points
    bullets = re.findall(r'^[\-\*\u2022]\s+(.*)', text, flags=re.MULTILINE)
    if bullets:
        return [s.strip() for s in bullets if s.strip()]

    # Strategy 3: Logic shift keywords
    keywords = [
        "However", "But", "Also", "In addition", "Moreover", "Nevertheless", "Then", "Finally", "Afterwards"
    ]
    pattern_logic = '|'.join([re.escape(k) for k in keywords])
    segments = re.split(rf'(?:^|\n)\s*(?:{pattern_logic})', text, flags=re.IGNORECASE)
    segments = [s.strip() for s in segments if len(s.strip()) > 20]
    if segments: return segments

    # Fallback: Split by newlines
    return [line.strip() for line in text.splitlines() if line.strip()]
