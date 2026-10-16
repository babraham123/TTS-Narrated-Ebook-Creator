"""Spell out numbers and the symbols around them before TTS, in the book's language.

The TTS cleaner deletes symbols the models stumble over, which turned "50%" into "50"
and "$5" into "5", and Chatterbox Multilingual reads digits poorly. Symbols are replaced
with words and numbers are spelled out with num2words. Anything ambiguous (dates such as
"3.5.1990", version numbers, "1.5" in a comma-decimal language) is left as written.
Languages without a word table keep the old behaviour: digits stay and symbols are dropped.

This only ever sees the text sent to TTS; the book's XHTML keeps the author's text.
"""

import re

from num2words import num2words

# Stands in for a space inside a number ("12 345"); NFKC would turn a no-break space
# into an ordinary one, and an ordinary space is not a reliable digit-group separator.
GROUP_MARK = ""

# Word forms per language. Units are (singular, plural[, one]): the singular is used only
# when the amount is exactly 1, and `one` replaces the language's default "one" before
# that unit ("ein Euro", "una libra"). Cue words are lowercase; a four-digit number after
# one of them is read as a year.
WORDS = {
    "en": {
        "percent": "percent", "degree": ("degree", "degrees"), "and": "and",
        "$": ("dollar", "dollars"), "€": ("euro", "euros"), "£": ("pound", "pounds"),
        "¥": ("yen", "yen"), "number": "number",
        "point": "point", "to": "to", "minus": "minus",
        "year_cues": "in since by from until till to of after before around circa c. "
                     "year anno ad between and january february march april may june july "
                     "august september october november december",
        "chapter": "chapter part book volume act scene section appendix canto",
        "regnal": ("the {}", None),
    },
    "da": {
        "percent": "procent", "degree": ("grad", "grader"), "and": "og", "one": "en",
        "$": ("dollar", "dollar"), "€": ("euro", "euro"), "£": ("pund", "pund", "et"),
        "¥": ("yen", "yen"),
        "point": "komma", "to": "til", "minus": "minus",
        "year_cues": "i siden fra til år året efter før indtil omkring januar februar marts "
                     "april maj juni juli august september oktober november december",
        "chapter": "kapitel del bog bind akt scene afsnit",
        "regnal": ("den {}", None),
    },
    "de": {
        "percent": "Prozent", "degree": ("Grad", "Grad"), "and": "und", "one": "ein",
        "$": ("Dollar", "Dollar"), "€": ("Euro", "Euro"), "£": ("Pfund", "Pfund"),
        "¥": ("Yen", "Yen"),
        "point": "Komma", "to": "bis", "minus": "minus",
        "year_cues": "im seit bis von ab um nach vor jahr jahre jahres anno januar jänner "
                     "februar märz april mai juni juli august september oktober november "
                     "dezember",
        "chapter": "kapitel teil buch band akt szene abschnitt anhang",
        "regnal": ("der {}", None),
    },
    "es": {
        "percent": "por ciento", "degree": ("grado", "grados"), "and": "y", "one": "un",
        "$": ("dólar", "dólares"), "€": ("euro", "euros"), "£": ("libra", "libras", "una"),
        "¥": ("yen", "yenes"),
        "point": "coma", "to": "a", "minus": "menos",
        "year_cues": "en desde hasta hacia de año",
        "chapter": "capítulo parte libro tomo acto escena sección",
        "regnal": ("{}", 10),
    },
    "fr": {
        "percent": "pour cent", "degree": ("degré", "degrés"), "and": "et",
        "$": ("dollar", "dollars"), "€": ("euro", "euros"), "£": ("livre", "livres", "une"),
        "¥": ("yen", "yens"),
        "point": "virgule", "to": "à", "minus": "moins",
        "year_cues": "en depuis dès vers avant après année an",
        "chapter": "chapitre partie livre tome acte scène section",
        "regnal": ("{}", 1),
    },
    "it": {
        "percent": "per cento", "degree": ("grado", "gradi"), "and": "e", "one": "un",
        "$": ("dollaro", "dollari"), "€": ("euro", "euro"),
        "£": ("sterlina", "sterline", "una"), "¥": ("yen", "yen"),
        "point": "virgola", "to": "a", "minus": "meno",
        "year_cues": "nel dal al del entro verso anno",
        "chapter": "capitolo parte libro volume atto scena sezione",
        "regnal": ("{}", None),
    },
    "nl": {
        "percent": "procent", "degree": ("graad", "graden"), "and": "en",
        "$": ("dollar", "dollar"), "€": ("euro", "euro"), "£": ("pond", "pond"),
        "¥": ("yen", "yen"),
        "point": "komma", "to": "tot", "minus": "min",
        "year_cues": "in sinds tot van vanaf na voor rond jaar januari februari maart april "
                     "mei juni juli augustus september oktober november december",
        "chapter": "hoofdstuk deel boek akte scène sectie bijlage",
        "regnal": ("de {}", None),
    },
    "no": {
        "percent": "prosent", "degree": ("grad", "grader"), "and": "og", "one": "en",
        "$": ("dollar", "dollar"), "€": ("euro", "euro"), "£": ("pund", "pund", "ett"),
        "¥": ("yen", "yen"),
        "point": "komma", "to": "til", "minus": "minus",
        "year_cues": "i siden fra til år året etter før innen rundt januar februar mars "
                     "april mai juni juli august september oktober november desember",
        "chapter": "kapittel del bok bind akt scene avsnitt",
        "regnal": ("den {}", None),
    },
    "pt": {
        "percent": "por cento", "degree": ("grau", "graus"), "and": "e",
        "$": ("dólar", "dólares"), "€": ("euro", "euros"), "£": ("libra", "libras", "uma"),
        "¥": ("iene", "ienes"),
        "point": "vírgula", "to": "a", "minus": "menos",
        "year_cues": "em desde até de ano",
        "chapter": "capítulo parte livro tomo ato cena secção seção",
        "regnal": ("{}", 10),
    },
    "sv": {
        "percent": "procent", "degree": ("grad", "grader"), "and": "och", "one": "en",
        "$": ("dollar", "dollar"), "€": ("euro", "euro"), "£": ("pund", "pund", "ett"),
        "¥": ("yen", "yen"),
        "point": "komma", "to": "till", "minus": "minus",
        "year_cues": "i sedan från till år året efter före innan omkring januari februari "
                     "mars april maj juni juli augusti september oktober november december",
        "chapter": "kapitel del bok band akt scen avsnitt",
        "regnal": ("den {}", None),
    },
}

# num2words slips, fixed after the fact.
_FIXUPS = {
    "da": [(re.compile(r"\ben millioner\b"), "en million")],
    "nl": [(re.compile(r"(?<=\w)één"), "een")],
    "sv": [(re.compile("etttusen"), "ettusen")],
}
# The feminine "one", for hours ("une heure", "la una").
_FEMININE = {"fr": (r"\bun$", "une"), "es": (r"uno$", "una"), "it": (r"uno$", "una"),
             "pt": (r"\bum$", "uma")}

# English scale words after an amount ("$5 million", "$5m") go before the currency.
_EN_SCALES = {"k": "thousand", "m": "million", "bn": "billion",
              "thousand": "thousand", "million": "million",
              "billion": "billion", "trillion": "trillion"}

# Monarchs and popes whose numeral is read as an ordinal ("Henry VIII"). A list, because
# "Then I" or "World War II" must not become "Then the first" or "World War the second".
_REGNAL_NAMES = set("""
    Henry Charles Elizabeth Edward George William James Louis Mary Richard John Stephen
    Victoria Anne Peter Paul Pius Leo Gregory Benedict Innocent Clement Urban Boniface
    Sixtus Alexander Nicholas Catherine Philip Frederick Ferdinand Isabella Napoleon Ivan
    Rudolf Maximilian Leopold Franz Joseph Josef Ramesses Ramses Selim Mehmed Suleiman
    Heinrich Friedrich Wilhelm Karl Ludwig Otto Maria Elisabeth Katharina Philipp
    Henri François Jean Philippe Napoléon Juan Carlos Felipe Luis Enrique Fernando Isabel
    Alfonso Sancho Jaime Pedro João Manuel Afonso Carlo Vittorio Emanuele Emmanuel Umberto
    Giovanni Luigi
    Enrico Federico Gustav Gustaf Christian Frederik Haakon Olav Harald Magnus Margrethe
    Carl Oscar Oskar Erik Kristian Willem Beatrix Juliana Wilhelmina Robert David Malcolm
""".split())
_FEMALE_NAMES = {"Elizabeth", "Victoria", "Anne", "Mary", "Catherine", "Isabella", "Maria",
                 "Elisabeth", "Katharina", "Isabel", "Margrethe", "Beatrix", "Juliana",
                 "Wilhelmina"}
_REGNAL = re.compile(
    rf"\b(?P<name>{'|'.join(sorted(_REGNAL_NAMES, key=len, reverse=True))})"
    # German and Scandinavian write "Heinrich VIII. war"; that dot is not a full stop.
    r"\s+(?P<num>[IVXLC]+)(?!\w)(?:\.(?=\s+[a-zäöüßæøå]))?")
# A lone "I" needs a clear end ("Part I." / "Part I:"), so "Part I think" is left alone.
_CHAPTER = {
    lang: re.compile(
        rf"\b(?P<kw>{'|'.join(words['chapter'].split())})\s+"
        r"(?P<num>[IVXLCDM]{2,}|I(?=\s*(?:$|[.:;,)–—-])))(?![\w'])",
        re.IGNORECASE)
    for lang, words in WORDS.items()
}
_ROMAN_HEADING = re.compile(r"\s*([IVXL]{2,})\.?\s*")
# "XIXe siècle", "Ier"; two letters at least, since "Le" and "Ce" are words.
_FR_ROMAN_ORDINAL = re.compile(r"\b(?P<num>[IVXLC]{2,}|I)(?P<sfx>er|re|e|ème)\b")
_ROMAN = re.compile(r"M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})")
_ROMAN_VALUES = {"M": 1000, "D": 500, "C": 100, "L": 50, "X": 10, "V": 5, "I": 1}

_DE_MONTHS = set("januar jänner februar märz april mai juni juli august september oktober "
                 "november dezember".split())
_DE_DATIVE = {"am", "im", "vom", "zum", "zur", "beim", "dem", "den", "des"}
_DE_DETERMINERS = _DE_DATIVE | {"der", "die", "das", "ein", "eine", "einen", "bis"}

# A number: digits joined by single separators. Whether a separator groups thousands or
# marks decimals depends on the language; a number that fits neither is left as written.
_NUM = rf"\d+(?:[.,{GROUP_MARK}]\d+)*"
_START = rf"(?<![\w.,{GROUP_MARK}])"
_END = rf"(?![\w{GROUP_MARK}]|[.,]\d)"
_SPACE_GROUPS = re.compile(r"(?<![\d.,])\d{1,3}(?: \d{3})+(?!\d)")
_PREFIX_CURRENCY = re.compile(
    rf"(?P<sym>[$€£¥])\s?(?P<num>{_NUM})"
    r"(?:\s?(?P<scale>thousand|million|billion|trillion|bn|k|m)\b)?",
    re.IGNORECASE,
)
_SUFFIX_CURRENCY = re.compile(rf"{_START}(?P<num>{_NUM})\s?(?P<sym>[$€£¥])")
_PERCENT = re.compile(rf"{_START}(?P<num>{_NUM})\s?%")
# After NFKC a masculine ordinal sign used as a degree sign ("20º C") reads "20o C".
_DEGREES = re.compile(
    rf"(?:(?<![\w.,])(?P<minus>[-−]))?{_START}(?P<num>{_NUM})\s?"
    r"(?:°|o(?=\s?[CF](?![A-Za-z])))(?:\s?(?P<unit>[CF])(?![A-Za-z]))?")
_AMPERSAND = re.compile(r"\s+&\s+")
_HASH_NUMBER = re.compile(r"#(?=\d)")
_PHONE = re.compile(r"(?<![\w+])\+\d[\d ()-]{5,}\d(?!\w)")
_DIGIT_CHAIN = re.compile(r"(?<![\w.,-])\d+(?:-\d+){2,}(?![\w-])")
_RANGE = re.compile(rf"{_START}(?P<a>\d+)\s?(?P<dash>[–-])\s?(?P<b>\d+){_END}")
# A trailing "Uhr"/"uur" is consumed: the spoken time already says it.
_TIME = re.compile(r"(?<![\w:.,])(?P<h>[01]?\d|2[0-3]):(?P<m>[0-5]\d)(?![\w:]|[.,]\d)"
                   r"(?:\s(?:Uhr|uur)\b)?")
_FR_TIME = re.compile(r"(?<![\w:.,])(?P<h>[01]?\d|2[0-3])\s?h(?:\s?(?P<m>[0-5]\d))?(?!\w)")
_EN_DECADE = re.compile(r"(?<![\w'])'?(?P<num>\d{1,3}0)s\b")
_EN_ERA = re.compile(r"\s*(?:AD|BC|BCE|CE)\b")
_BARE = re.compile(rf"{_START}(?P<num>{_NUM}){_END}")
_ORDINALS = {
    "en": re.compile(rf"{_START}(?P<num>\d+)(?P<sfx>st|nd|rd|th)\b"),
    "fr": re.compile(rf"{_START}(?P<num>\d+)(?P<sfx>er|re|ère|e|ème)(?P<pl>s?)\b"),
    # º and ª are "o" and "a" after NFKC.
    "es": re.compile(rf"{_START}(?P<num>\d+)\.?(?P<sfx>[oa])(?P<pl>s?)\b(?!\s?[CF]\b)"),
    "nl": re.compile(rf"{_START}(?P<num>\d+)(?P<sfx>ste|de|e)\b"),
    "sv": re.compile(rf"{_START}(?P<num>\d+):(?P<sfx>[ae])\b"),
    # "3." is an ordinal only when the context says so; see _dotted_ordinal.
    "de": re.compile(rf"{_START}(?P<num>\d+)\.(?=\s+(?P<next>\w+))"),
}
_ORDINALS["it"] = _ORDINALS["pt"] = _ORDINALS["es"]
_ORDINALS["da"] = _ORDINALS["no"] = _ORDINALS["de"]


def n2w(value, lang, to="cardinal"):
    """num2words for `lang` without its commas (pauses in TTS) and known slips."""
    words = num2words(value, lang=lang, to=to).replace(",", "")
    for pattern, fixed in _FIXUPS.get(lang, ()):
        words = pattern.sub(fixed, words)
    return words


def _parse(num, lang):
    """`num` as (integer, fraction digits or None) in `lang`'s notation, or None."""
    group, point = (",", ".") if lang == "en" else (".", ",")
    m = re.fullmatch(rf"(\d{{1,3}}(?:[{group}{GROUP_MARK}]\d{{3}})+|\d+)(?:\{point}(\d+))?", num)
    if m is None:
        return None
    return int(re.sub(r"\D", "", m.group(1))), m.group(2)


def _digits(text, lang):
    return " ".join(n2w(int(d), lang) for d in text if d.isdigit())


def _spell(num, lang):
    """Words for `num`, or None when its notation is ambiguous in `lang`. Decimals are
    read digit by digit after the decimal word ("three point one four")."""
    parsed = _parse(num, lang)
    if parsed is None:
        return None
    whole, fraction = parsed
    if whole >= 10 ** 15:
        return _digits(num, lang)
    words = n2w(whole, lang)
    if fraction:
        words += f" {WORDS[lang]['point']} {_digits(fraction, lang)}"
    return words


def _year(value, lang):
    if lang == "sv":  # num2words has no Swedish years: "nittonhundraåttiofyra"
        if 1100 <= value < 2000:
            rest = value % 100
            return n2w(value // 100, lang) + "hundra" + (n2w(rest, lang) if rest else "")
        return n2w(value, lang)
    return n2w(value, lang, to="year")


def _is_year(num):
    return len(num) == 4 and num.isdigit() and 1000 <= int(num) <= 2099


def _feminine(words, lang):
    pattern, fem = _FEMININE.get(lang, (None, None))
    return re.sub(pattern, fem, words) if pattern else words


def _with_unit(num, lang, forms):
    """('5', ('dollar', 'dollars')) -> 'five dollars'; digits stay if `num` is ambiguous."""
    parsed = _parse(num, lang)
    if parsed is None:
        return f"{num} {forms[1]}"
    if parsed == (1, None):
        one = forms[2] if len(forms) > 2 else WORDS[lang].get("one", n2w(1, lang))
        return f"{one} {forms[0]}"
    return f"{_spell(num, lang)} {forms[1]}"


def _previous_word(text, end):
    m = re.search(r"(\S+)\s+$", text[:end])
    return m.group(1).strip("(\"'«»“”‘’").lower() if m else ""


def _roman_value(numeral):
    if not numeral or not _ROMAN.fullmatch(numeral):
        return None
    values = [_ROMAN_VALUES[c] for c in numeral]
    return sum(-v if i + 1 < len(values) and v < values[i + 1] else v
               for i, v in enumerate(values))


def _regnal(value, name, lang):
    template, ordinal_up_to = WORDS[lang]["regnal"]
    if ordinal_up_to is not None and value > ordinal_up_to:
        return n2w(value, lang)  # "Louis quatorze", "Alfonso trece"
    words = n2w(value, lang, to="ordinal")
    female = name in _FEMALE_NAMES
    if lang == "de":
        words = words.capitalize()
        template = "die {}" if female else template
    elif female and lang == "fr":
        words = "première"
    elif female and lang in ("es", "it", "pt"):
        words = re.sub(r"o\b", "a", words)
    return template.format(words)


def _roman_numerals(text, lang):
    heading = _ROMAN_HEADING.fullmatch(text)
    if heading and _roman_value(heading.group(1)):
        return n2w(_roman_value(heading.group(1)), lang)

    def chapter(m):
        num = m.group("num")
        value = _roman_value(num) if num.isupper() else None
        return f"{m.group('kw')} {n2w(value, lang)}" if value else m.group(0)

    def regnal(m):
        value = _roman_value(m.group("num"))
        return f"{m.group('name')} {_regnal(value, m.group('name'), lang)}" if value \
            else m.group(0)

    def fr_ordinal(m):
        value = _roman_value(m.group("num"))
        if not value:
            return m.group(0)
        return "première" if m.group("sfx") == "re" and value == 1 \
            else n2w(value, lang, to="ordinal")

    text = _CHAPTER[lang].sub(chapter, text)
    text = _REGNAL.sub(regnal, text)
    if lang == "fr":
        text = _FR_ROMAN_ORDINAL.sub(fr_ordinal, text)
    return text


def _digit_runs(text, lang):
    """Phone numbers and ISBN-like chains are read digit by digit, a pause between groups."""
    def by_groups(m):
        return ", ".join(_digits(g, lang) for g in re.findall(r"\d+", m.group(0)))

    return _DIGIT_CHAIN.sub(by_groups, _PHONE.sub(by_groups, text))


def _symbols(text, lang):
    words = WORDS[lang]

    def money(num, forms):
        # Two decimals are minor units: "£2.50" -> "two pounds fifty".
        parsed = _parse(num, lang)
        if parsed and parsed[1] and len(parsed[1]) == 2:
            spoken = _with_unit(str(parsed[0]), lang, forms)
            cents = int(parsed[1])
            return f"{spoken} {n2w(cents, lang)}" if cents else spoken
        return _with_unit(num, lang, forms)

    def prefix_currency(m):
        scale = (m.group("scale") or "").lower()
        forms = words[m.group("sym")]
        if not scale:
            return money(m.group("num"), forms)
        if lang != "en":
            return m.group(0)  # word order differs ("5 millions de dollars"); leave it
        return f"{_spell(m.group('num'), lang) or m.group('num')} {_EN_SCALES[scale]} {forms[1]}"

    def degrees(m):
        num, forms = m.group("num"), words["degree"]
        if m.group("minus"):
            spoken = f"{words['minus']} {_spell(num, lang) or num} {forms[1]}"
        else:
            spoken = _with_unit(num, lang, forms)
        return spoken + {"C": " Celsius", "F": " Fahrenheit"}.get(m.group("unit") or "", "")

    text = _PREFIX_CURRENCY.sub(prefix_currency, text)
    text = _SUFFIX_CURRENCY.sub(lambda m: money(m.group("num"), words[m.group("sym")]), text)
    text = _PERCENT.sub(lambda m: _with_unit(m.group("num"), lang, (words["percent"],) * 2), text)
    text = _DEGREES.sub(degrees, text)
    text = _AMPERSAND.sub(f" {words['and']} ", text)
    if "number" in words:
        text = _HASH_NUMBER.sub(f"{words['number']} ", text)
    return text


def _ranges(text, lang):
    """'1914–1918' -> years; '12–15' -> 'twelve to fifteen'. A hyphen counts only between
    ascending numbers that don't look like a local phone number; '555-1234' or a score
    '3-2' is read digit by digit."""
    to = WORDS[lang]["to"]

    def sub(m):
        a, b = m.group("a"), m.group("b")
        if _is_year(a) and len(b) == 2 and int(b) > int(a) % 100:  # "1914–18"
            return f"{_year(int(a), lang)} {to} {n2w(int(b), lang)}"
        if _is_year(a) and _is_year(b) and int(a) < int(b):
            return f"{_year(int(a), lang)} {to} {_year(int(b), lang)}"
        phone = (len(a), len(b)) == (3, 4)
        if m.group("dash") == "–" or (int(a) < int(b) and len(b) <= 4 and not phone):
            return f"{n2w(int(a), lang)} {to} {n2w(int(b), lang)}"
        return f"{_digits(a, lang)}, {_digits(b, lang)}"

    return _RANGE.sub(sub, text)


def _clock(h, minutes, lang):
    hour, m = n2w(h, lang), int(minutes or 0)
    mins = n2w(m, lang) if m else ""
    if lang == "en":
        if not m:
            return f"{hour} o'clock"
        return f"{hour} oh {mins}" if m < 10 else f"{hour} {mins}"
    if lang == "de":
        return f"{'ein' if h == 1 else hour} Uhr {mins}".strip()
    if lang == "nl":
        return f"{hour} uur {mins}".strip()
    hour = _feminine(hour, lang)
    if lang == "fr":
        return f"{hour} {'heure' if h <= 1 else 'heures'} {mins}".strip()
    joiner = {"es": " y ", "it": " e ", "pt": " e "}.get(lang, " ")
    return f"{hour}{joiner}{mins}" if m else hour


def _times(text, lang):
    text = _TIME.sub(lambda m: _clock(int(m.group("h")), m.group("m"), lang), text)
    if lang == "fr":
        text = _FR_TIME.sub(lambda m: _clock(int(m.group("h")), m.group("m"), lang), text)
    return text


def _dotted_ordinal(m, text, lang):
    """de/da/no "3." is an ordinal before a lowercase word ("3. plass", "3. großen") or,
    in German, before a month or after an article ("am 3. Mai", "der 2. Weltkrieg"). At
    the end of a sentence, or before a capitalised word, it stays a cardinal."""
    nxt, prev = m.group("next"), _previous_word(text, m.start())
    clear = nxt[0].islower() or (lang == "de" and (nxt.lower() in _DE_MONTHS
                                                   or prev in _DE_DETERMINERS))
    if not clear:
        return m.group(0)
    words = n2w(int(m.group("num")), lang, to="ordinal")
    return words + "n" if lang == "de" and prev in _DE_DATIVE else words  # "am dritten"


def _ordinals(text, lang):
    def sub(m):
        if lang in ("de", "da", "no"):
            return _dotted_ordinal(m, text, lang)
        value, sfx = int(m.group("num")), m.group("sfx")
        words = n2w(value, lang, to="ordinal")
        if lang == "fr" and sfx in ("re", "ère") and value == 1:
            words = "première"
        elif lang in ("es", "it", "pt") and sfx == "a":
            words = " ".join(re.sub(r"o$", "a", w) for w in words.split())
        return words + ("s" if m.groupdict().get("pl") else "")

    return _ORDINALS[lang].sub(sub, text)


def _decades(text):
    def sub(m):
        num = m.group("num")
        words = _year(int(num), "en") if len(num) == 4 else n2w(int(num), "en")
        return words[:-1] + "ies" if words.endswith("y") else words + "s"

    return _EN_DECADE.sub(sub, text)


def _bare_numbers(text, lang):
    """Cardinals and decimals; a four-digit number in 1000-2099 after a cue word, in
    parentheses, or standing alone before punctuation is a year."""
    cues = set(WORDS[lang]["year_cues"].split())

    def sub(m):
        num = m.group("num")
        if _is_year(num):
            after = text[m.end():]
            if (_previous_word(text, m.start()) in cues
                    or re.match(r"\s*(?:$|[.,;:!?)\]])", after)
                    or (lang == "en" and _EN_ERA.match(after))):
                return _year(int(num), lang)
        if re.fullmatch(r"0\d+", num) or (num.isdigit() and len(num) >= 7):
            return _digits(num, lang)  # an identifier rather than an amount
        return _spell(num, lang) or num

    return _BARE.sub(sub, text)


def normalize_symbols(text, lang="en"):
    """Numbers and symbols as words in `lang`; unchanged for a language without a table."""
    if lang not in WORDS:
        return text
    if lang != "en":
        text = _SPACE_GROUPS.sub(lambda m: m.group(0).replace(" ", GROUP_MARK), text)
    text = _roman_numerals(text, lang)
    text = _digit_runs(text, lang)
    text = _symbols(text, lang)
    text = _ranges(text, lang)
    text = _times(text, lang)
    text = _ordinals(text, lang)
    if lang == "en":
        text = _decades(text)
    return _bare_numbers(text, lang).replace(GROUP_MARK, " ")
