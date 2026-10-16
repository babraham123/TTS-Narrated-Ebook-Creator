"""Numbers and symbols are spoken in the book's language; the book's own text is untouched."""

import pytest
from num2words import CONVERTER_CLASSES, num2words

from calliope.epub import clean_text_for_tts, process_xhtml_inplace
from calliope.normalize import WORDS
from calliope.qa import error_rate

LANGS = sorted(WORDS)


def spoken(text, lang):
    return clean_text_for_tts(text, lang)


def test_every_language_with_a_word_table_is_covered_by_num2words():
    assert LANGS == ["da", "de", "en", "es", "fr", "it", "nl", "no", "pt", "sv"]
    assert set(LANGS) <= set(CONVERTER_CLASSES)


@pytest.mark.parametrize("lang, text, expected", [
    ("en", "Prices rose 50% to $5.", "Prices rose fifty percent to five dollars."),
    ("en", "It cost $1, then £2.50.", "It cost one dollar, then two pounds fifty."),
    ("en", "A $5m deal and a $2.5 billion fund.",
     "A five million dollars deal and a two point five billion dollars fund."),
    ("en", "It was 20°C, or 68 °F, and 1° warmer; -5 °C at night.",
     "It was twenty degrees Celsius, or sixty-eight degrees Fahrenheit, and one degree warmer; "
     "minus five degrees Celsius at night."),
    ("en", "Tom & Jerry were the #1 show.", "Tom and Jerry were the number one show."),
    ("en", "Fullwidth ５０％ and ℃ fold first: 5℃.",
     "Fullwidth fifty percent and C fold first: five degrees Celsius."),
    ("de", "Es kostet 1 €, dann 2,50 € und 20 % mehr.",
     "Es kostet ein Euro, dann zwei Euro fünfzig und zwanzig Prozent mehr."),
    ("fr", "Il fait 30 °C & ça coûte 1 € ou 1 £.",
     "Il fait trente degrés Celsius et ça coûte un euro ou une livre."),
    ("es", "Sube un 3 % hasta $10, a 20º C.",
     "Sube un tres por ciento hasta diez dólares, a veinte grados Celsius."),
    ("no", "Det er 2 grader, eller 2°, og 10 % rabatt.",
     "Det er to grader, eller to grader, og ti prosent rabatt."),
    ("sv", "Det kostar 1 £ och 1 $.", "Det kostar ett pund och en dollar."),
    # Without a word table the symbol is still dropped and digits stay, as before.
    ("ja", "50%です", "50です"),
])
def test_symbols_are_spoken(lang, text, expected):
    assert spoken(text, lang) == expected


def test_other_symbols_are_still_removed():
    assert spoken("A *bold* claim [1] → yes", "en") == "A bold claim one yes"


@pytest.mark.parametrize("lang", LANGS)
def test_cardinals_and_decimals_use_the_languages_separators(lang):
    point = WORDS[lang]["point"]
    # num2words writes Swedish "etttusen"; the word is "ettusen".
    thousands = num2words(1250, lang=lang).replace(",", "").replace("etttusen", "ettusen")
    expected = f"{thousands} {point} {num2words(5, lang=lang)}"
    if lang == "en":
        assert spoken("1,250.5", lang) == expected
        assert spoken("1.250,5", lang) == "1.250,5"  # not English notation: left alone
    else:
        assert spoken("1.250,5", lang) == expected
        assert spoken("1\N{NO-BREAK SPACE}250,5", lang) == expected
        assert spoken("1 250,5", lang) == expected
        assert spoken("1,250.5", lang) == "1,250.5"
    assert spoken("42", lang) == num2words(42, lang=lang)


@pytest.mark.parametrize("lang, text, expected", [
    ("en", "The 1st, 2nd, 3rd and 21st place.", "The first, second, third and twenty-first place."),
    ("de", "Am 3. Mai kam der 2. Zug zum 1. großen Fest.",
     "Am dritten Mai kam der zweite Zug zum ersten großen Fest."),
    ("de", "Er kam um 3. Dann ging er.", "Er kam um drei. Dann ging er."),
    ("da", "Det var den 3. dag, den 1. maj.", "Det var den tredje dag, den første maj."),
    ("no", "Han kom på 3. plass. Hun kom klokka 3.",
     "Han kom på tredje plass. Hun kom klokka tre."),
    ("fr", "Le 1er mai, la 1re fois, le 2e jour, les 3es places, le XIXe siècle.",
     "Le premier mai, la première fois, le deuxième jour, les troisièmes places, "
     "le dix-neuvième siècle."),
    ("es", "El 1º y la 2ª vez, la 3.ª planta.", "El primero y la segunda vez, la tercera planta."),
    ("it", "Il 1º e la 2ª volta.", "Il primo e la seconda volta."),
    ("pt", "O 1º e a 2ª vez.", "O primeiro e a segunda vez."),
    ("nl", "De 1e, 3de en 8ste keer.", "De eerste, derde en achtste keer."),
    ("sv", "Den 1:a och 3:e gången.", "Den första och tredje gången."),
])
def test_ordinals(lang, text, expected):
    assert spoken(text, lang) == expected


@pytest.mark.parametrize("lang, cue", [
    ("en", "in"), ("da", "i"), ("de", "im Jahr"), ("es", "en"), ("fr", "en"), ("it", "nel"),
    ("nl", "in"), ("no", "i"), ("pt", "em"),
])
def test_years_after_a_cue_word(lang, cue):
    year = num2words(1984, lang=lang, to="year")
    assert spoken(f"{cue} 1984 kam", lang) == f"{cue} {year} kam"


@pytest.mark.parametrize("text, expected", [
    ("In 1984 she left.", "In nineteen eighty-four she left."),
    ("Orwell wrote it in (1949), not 1948.", "Orwell wrote it in nineteen forty-nine, not nineteen forty-eight."),
    ("It fell in 1066 AD, since 2005.", "It fell in ten sixty-six AD, since two thousand and five."),
    ("There were 1500 men.", "There were one thousand five hundred men."),
    ("In the 1990s and the '80s.", "In the nineteen nineties and the eighties."),
])
def test_english_years(text, expected):
    assert spoken(text, "en") == expected


def test_swedish_years_are_spelled_by_hundreds():
    # num2words has no Swedish years; Swedish reads 1984 as "nittonhundraåttiofyra".
    assert spoken("Sedan 1984 och 2005.", "sv") == "Sedan nittonhundraåttiofyra och tvåtusenfem."


@pytest.mark.parametrize("lang, text, expected", [
    ("en", "Chapter XII. Part I: Henry VIII and Elizabeth I.",
     "Chapter twelve. Part one: Henry the eighth and Elizabeth the first."),
    ("en", "Then I left. World War II. Part I think.", "Then I left. World War II. Part I think."),
    ("en", "IV", "four"),
    ("en", "I", "I"),
    ("de", "Kapitel IV: Heinrich VIII. war König.", "Kapitel vier: Heinrich der Achte war König."),
    ("fr", "Chapitre III : Louis XIV et François Ier.", "Chapitre trois : Louis quatorze et François premier."),
    ("es", "Capítulo IV: Felipe II y Alfonso XIII.", "Capítulo cuatro: Felipe segundo y Alfonso trece."),
    ("it", "Capitolo II: Vittorio Emanuele II.", "Capitolo due: Vittorio Emanuele secondo."),
    ("pt", "Pedro II e João VI.", "Pedro segundo e João sexto."),
    ("nl", "Hoofdstuk XII: Willem III.", "Hoofdstuk twaalf: Willem de derde."),
    ("sv", "Kapitel II: Gustav III.", "Kapitel två: Gustav den tredje."),
    ("no", "Haakon VII og Olav V.", "Haakon den syvende og Olav den femte."),
    ("da", "Christian IV.", "Christian den fjerde."),
])
def test_roman_numerals_in_clear_contexts(lang, text, expected):
    assert spoken(text, lang) == expected


@pytest.mark.parametrize("lang, text, expected", [
    ("en", "The war of 1914–1918, or 1939-45.",
     "The war of nineteen fourteen to nineteen eighteen, or nineteen thirty-nine to forty-five."),
    ("en", "See pages 12-15.", "See pages twelve to fifteen."),
    ("de", "Von 1914–1918.", "Von neunzehnhundertvierzehn bis neunzehnhundertachtzehn."),
    ("en", "At 10:30, 7:05 and 9:00.", "At ten thirty, seven oh five and nine o'clock."),
    ("de", "Um 10:30 Uhr.", "Um zehn Uhr dreißig."),
    ("fr", "À 1:15 ou 10h30.", "À une heure quinze ou dix heures trente."),
    ("es", "A la 1:30.", "A la una y treinta."),
    ("en", "Call 555-1234 or +1 555 123 4567.",
     "Call five five five, one two three four or one, five five five, one two three, "
     "four five six seven."),
    ("en", "ISBN 978-3-16-148410-0.",
     "ISBN nine seven eight, three, one six, one four eight four one zero, zero."),
    ("en", "Agent 007, account 12345678.",
     "Agent zero zero seven, account one two three four five six seven eight."),
])
def test_ranges_times_and_digit_runs(lang, text, expected):
    assert spoken(text, lang) == expected


@pytest.mark.parametrize("lang, text", [
    ("en", "Version 1.2.3 is out."),
    ("de", "Am 3.5.1990 um 1.5 Uhr."),
    ("en", "A 3D film in 10km."),
])
def test_ambiguous_numbers_are_left_alone(lang, text):
    assert spoken(text, lang) == text


def test_only_the_tts_text_is_normalised(tmp_path):
    page = tmp_path / "ch1.xhtml"
    page.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>T</title></head>'
        "<body><p>In 1984 she paid $5. Henry VIII came 2nd.</p></body></html>",
        encoding="utf-8")

    segments, _ = process_xhtml_inplace(str(page), 0, "style.css", "en")

    assert [s["text"] for s in segments] == [
        "In nineteen eighty-four she paid five dollars.", "Henry the eighth came second."]
    xhtml = page.read_text(encoding="utf-8")
    assert "In 1984 she paid $5. </span>" in xhtml
    assert "Henry VIII came 2nd.</span>" in xhtml


@pytest.mark.parametrize("lang, reference, transcript", [
    ("en", "Chapter One", "Chapter 1."),
    ("en", spoken("In 1984 it cost $5.", "en"), "In 1984 it cost $5."),
    ("de", spoken("Am 3. Mai 1984.", "de"), "Am 3. Mai 1984."),
])
def test_whisper_digits_match_spelled_out_references(lang, reference, transcript):
    assert error_rate(reference, transcript, lang) == 0
