"""Spell out symbols that carry meaning before TTS, in the book's language.

The TTS cleaner deletes symbols the models stumble over, which turned "50%" into "50"
and "$5" into "5". The symbols below are replaced with words first. Languages without
a word table keep the old behaviour: the symbol is dropped.
"""

import re

# Word forms per language. Units are (singular, plural); the singular is used only
# when the number is exactly 1.
WORDS = {
    "en": {"percent": "percent", "degree": ("degree", "degrees"), "and": "and",
           "$": ("dollar", "dollars"), "€": ("euro", "euros"), "£": ("pound", "pounds"),
           "¥": ("yen", "yen"), "number": "number"},
    "da": {"percent": "procent", "degree": ("grad", "grader"), "and": "og",
           "$": ("dollar", "dollar"), "€": ("euro", "euro"), "£": ("pund", "pund"),
           "¥": ("yen", "yen")},
    "de": {"percent": "Prozent", "degree": ("Grad", "Grad"), "and": "und",
           "$": ("Dollar", "Dollar"), "€": ("Euro", "Euro"), "£": ("Pfund", "Pfund"),
           "¥": ("Yen", "Yen")},
    "es": {"percent": "por ciento", "degree": ("grado", "grados"), "and": "y",
           "$": ("dólar", "dólares"), "€": ("euro", "euros"), "£": ("libra", "libras"),
           "¥": ("yen", "yenes")},
    "fr": {"percent": "pour cent", "degree": ("degré", "degrés"), "and": "et",
           "$": ("dollar", "dollars"), "€": ("euro", "euros"), "£": ("livre", "livres"),
           "¥": ("yen", "yens")},
    "it": {"percent": "per cento", "degree": ("grado", "gradi"), "and": "e",
           "$": ("dollaro", "dollari"), "€": ("euro", "euro"), "£": ("sterlina", "sterline"),
           "¥": ("yen", "yen")},
    "nl": {"percent": "procent", "degree": ("graad", "graden"), "and": "en",
           "$": ("dollar", "dollar"), "€": ("euro", "euro"), "£": ("pond", "pond"),
           "¥": ("yen", "yen")},
    "no": {"percent": "prosent", "degree": ("grad", "grader"), "and": "og",
           "$": ("dollar", "dollar"), "€": ("euro", "euro"), "£": ("pund", "pund"),
           "¥": ("yen", "yen")},
    "pt": {"percent": "por cento", "degree": ("grau", "graus"), "and": "e",
           "$": ("dólar", "dólares"), "€": ("euro", "euros"), "£": ("libra", "libras"),
           "¥": ("iene", "ienes")},
    "sv": {"percent": "procent", "degree": ("grad", "grader"), "and": "och",
           "$": ("dollar", "dollar"), "€": ("euro", "euro"), "£": ("pund", "pund"),
           "¥": ("yen", "yen")},
}

# English scale words after an amount ("$5 million", "$5m") go before the currency.
_EN_SCALES = {"k": "thousand", "m": "million", "bn": "billion",
              "thousand": "thousand", "million": "million",
              "billion": "billion", "trillion": "trillion"}

_NUM = r"\d+(?:[.,  ]\d{3})*(?:[.,]\d+)?"
_PREFIX_CURRENCY = re.compile(
    rf"(?P<sym>[$€£¥])\s?(?P<num>{_NUM})"
    r"(?:\s?(?P<scale>thousand|million|billion|trillion|bn|k|m)\b)?",
    re.IGNORECASE,
)
_SUFFIX_CURRENCY = re.compile(rf"(?P<num>{_NUM})\s?(?P<sym>[$€£¥])")
_PERCENT = re.compile(rf"(?P<num>{_NUM})\s?%")
_DEGREES = re.compile(rf"(?P<num>{_NUM})\s?°(?:\s?(?P<unit>[CF])(?![A-Za-z]))?")
_AMPERSAND = re.compile(r"\s+&\s+")
_HASH_NUMBER = re.compile(r"#(?=\d)")


def _unit(forms, num):
    singular, plural = forms
    return singular if num == "1" else plural


def normalize_symbols(text, lang="en"):
    words = WORDS.get(lang)
    if words is None:
        return text

    def prefix_currency(m):
        scale = (m.group("scale") or "").lower()
        if scale and lang != "en":
            return m.group(0)  # word order differs ("5 millions de dollars"); leave it
        amount = m.group("num")
        if scale:
            return f"{amount} {_EN_SCALES[scale]} {words[m.group('sym')][1]}"
        return f"{amount} {_unit(words[m.group('sym')], amount)}"

    text = _PREFIX_CURRENCY.sub(prefix_currency, text)
    text = _SUFFIX_CURRENCY.sub(
        lambda m: f"{m.group('num')} {_unit(words[m.group('sym')], m.group('num'))}", text)
    text = _PERCENT.sub(lambda m: f"{m.group('num')} {words['percent']}", text)

    def degrees(m):
        unit = {"C": " Celsius", "F": " Fahrenheit"}.get(m.group("unit") or "", "")
        return f"{m.group('num')} {_unit(words['degree'], m.group('num'))}{unit}"

    text = _DEGREES.sub(degrees, text)
    text = _AMPERSAND.sub(f" {words['and']} ", text)
    if "number" in words:
        text = _HASH_NUMBER.sub(f"{words['number']} ", text)
    return text
