"""Symbols that carry meaning are spoken in the book's language instead of dropped."""

import pytest

from calliope.epub import clean_text_for_tts


@pytest.mark.parametrize("lang, text, spoken", [
    ("en", "Prices rose 50% to $5.", "Prices rose 50 percent to 5 dollars."),
    ("en", "It cost $1, then £2.50.", "It cost 1 dollar, then 2.50 pounds."),
    ("en", "A $5m deal and a $2 billion fund.", "A 5 million dollars deal and a 2 billion dollars fund."),
    ("en", "It was 20°C, or 68 °F, and 1° warmer.", "It was 20 degrees Celsius, or 68 degrees Fahrenheit, and 1 degree warmer."),
    ("en", "Tom & Jerry were the #1 show.", "Tom and Jerry were the number 1 show."),
    ("en", "Fullwidth ５０％ and ℃ fold first: 5℃.", "Fullwidth 50 percent and C fold first: 5 degrees Celsius."),
    ("de", "Es kostet 5 € und 20 % mehr.", "Es kostet 5 Euro und 20 Prozent mehr."),
    ("fr", "Il fait 30 °C & le prix est de 1 €.", "Il fait 30 degrés Celsius et le prix est de 1 euro."),
    ("es", "Sube un 3 % hasta $10.", "Sube un 3 por ciento hasta 10 dólares."),
    ("no", "Det er 2 grader, eller 2°, og 10 % rabatt.", "Det er 2 grader, eller 2 grader, og 10 prosent rabatt."),
    # Without a word table the symbol is still dropped, as before.
    ("ja", "50%です", "50です"),
])
def test_symbols_are_spoken(lang, text, spoken):
    assert clean_text_for_tts(text, lang) == spoken


def test_other_symbols_are_still_removed():
    assert clean_text_for_tts("A *bold* claim [1] → yes", "en") == "A bold claim 1 yes"
