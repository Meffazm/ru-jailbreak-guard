"""Pure text-filtering utilities used across data loaders."""

CYRILLIC_RANGE_LO = 0x0400
CYRILLIC_RANGE_HI = 0x04FF


def cyrillic_ratio(text: str) -> float:
    """Return the fraction of Cyrillic characters among all letters in text.

    Args:
        text: Input string.

    Returns:
        A float in [0.0, 1.0]. Empty input returns 0.0.
    """
    if not text:
        return 0.0
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    cyrillic = sum(1 for ch in letters if CYRILLIC_RANGE_LO <= ord(ch) <= CYRILLIC_RANGE_HI)
    return cyrillic / len(letters)


def is_russian_text(text: str, min_ratio: float = 0.5) -> bool:
    """True if Cyrillic-letter fraction meets min_ratio."""
    return cyrillic_ratio(text) >= min_ratio


def text_length_ok(text: str, min_chars: int = 10, max_chars: int = 4000) -> bool:
    """True if text length is within [min_chars, max_chars]."""
    return min_chars <= len(text) <= max_chars
