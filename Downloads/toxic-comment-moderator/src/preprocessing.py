import re

_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Lightweight, non-destructive normalization for toxicity detection.
    Replaces newlines/tabs with spaces so words never merge, while
    preserving casing patterns, punctuation, and obfuscated characters.
    """
    if not isinstance(text, str):
        return ""
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text