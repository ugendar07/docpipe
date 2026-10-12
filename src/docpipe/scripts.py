"""Unicode script ranges and share helpers. Add a script by adding its ranges here."""
RANGES = {
    "latin": [(0x41, 0x5A), (0x61, 0x7A), (0xC0, 0x24F), (0x1E00, 0x1EFF)],
    "devanagari": [(0x900, 0x97F)], "bengali": [(0x980, 0x9FF)], "gurmukhi": [(0xA00, 0xA7F)],
    "gujarati": [(0xA80, 0xAFF)], "oriya": [(0xB00, 0xB7F)], "tamil": [(0xB80, 0xBFF)],
    "telugu": [(0xC00, 0xC7F)], "kannada": [(0xC80, 0xCFF)], "malayalam": [(0xD00, 0xD7F)],
    "sinhala": [(0xD80, 0xDFF)], "thai": [(0xE00, 0xE7F)], "arabic": [(0x600, 0x6FF), (0x750, 0x77F)],
    "hebrew": [(0x590, 0x5FF)], "cyrillic": [(0x400, 0x52F)], "greek": [(0x370, 0x3FF)],
    "han": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)], "hangul": [(0xAC00, 0xD7AF)],
    "hiragana": [(0x3040, 0x309F)], "katakana": [(0x30A0, 0x30FF)],
}


def share(text, script):
    """Share of non-space characters that belong to `script`, and the character count."""
    c = [x for x in text if not x.isspace()]
    rs = RANGES[script]
    return (sum(any(a <= ord(x) <= b for a, b in rs) for x in c) / len(c) if c else 0.0), len(c)


def code(script, languages):
    return languages.get("codes", {}).get(script, script)
