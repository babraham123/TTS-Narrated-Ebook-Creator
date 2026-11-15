"""Sentence wrapping in process_xhtml_inplace: which text is narrated and the markup it leaves."""

import re

import pytest
from bs4 import BeautifulSoup

from calliope.epub import process_xhtml_inplace


def wrap(tmp_path, body):
    """Run process_xhtml_inplace on a chapter with this body; return (spoken texts, new XHTML)."""
    path = tmp_path / "chapter.xhtml"
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">'
        f"<head><title>T</title></head><body>{body}</body></html>",
        encoding="utf-8",
    )
    segments, _next_id = process_xhtml_inplace(str(path), 1, "sync.css")
    return [s["text"] for s in segments], path.read_text(encoding="utf-8")


def span_texts(xhtml):
    """Visible text of each sentence span, in document order."""
    soup = BeautifulSoup(xhtml, "xml")
    return [s.get_text() for s in soup.find_all("span", id=re.compile(r"^f\d{6}$"))]


@pytest.mark.parametrize(
    "body, spoken",
    [
        (
            "<ul><li>Outer item.<ul><li>Inner item.</li></ul>Outer tail.</li></ul>",
            ["Outer item.", "Inner item.", "Outer tail."],
        ),
        (
            "<div>Intro text.<p>A paragraph.</p>Trailing text.</div>",
            ["Intro text.", "A paragraph.", "Trailing text."],
        ),
        ("Loose in the body.<p>A paragraph.</p>", ["Loose in the body.", "A paragraph."]),
        ("<section>Section text.<h1>Title</h1></section>", ["Section text.", "Title"]),
        ("<dl><dt>Term</dt><dd>Its definition.</dd></dl>", ["Term", "Its definition."]),
        (
            "<figure><img src='a.png' alt=''/><figcaption>A caption.</figcaption></figure>",
            ["A caption."],
        ),
        ("<pre>Preformatted text.</pre>", ["Preformatted text."]),
    ],
    ids=["li", "div", "body", "section", "dl", "figcaption", "pre"],
)
def test_text_around_nested_blocks_is_narrated_in_reading_order(tmp_path, body, spoken):
    texts, xhtml = wrap(tmp_path, body)
    assert texts == spoken
    assert span_texts(xhtml) == spoken


def test_tables_are_skipped_but_their_caption_is_read(tmp_path):
    texts, xhtml = wrap(
        tmp_path,
        "<p>Before.</p><table><caption>Rainfall by month.</caption>"
        "<tr><th>Month</th></tr><tr><td>March</td><td><p>Wet.</p></td></tr></table>",
    )
    assert texts == ["Before.", "Rainfall by month."]
    assert "<td>March</td>" in xhtml
    assert "<td><p>Wet.</p></td>" in xhtml


def test_line_break_separates_words_and_is_kept(tmp_path):
    texts, xhtml = wrap(tmp_path, "<p>First line<br/>second line.</p>")
    assert texts == ["First line second line."]
    assert "<br/>" in xhtml
    assert span_texts(xhtml) == ["First line second line."]


def test_inline_element_split_across_sentences_keeps_its_ids_once(tmp_path):
    texts, xhtml = wrap(
        tmp_path,
        '<p>One <em id="e1" class="x">two. Three</em> four. '
        'Then <a name="n1" xml:id="x1" href="#n1">five. Six.</a></p>',
    )
    assert texts == ["One two.", "Three four.", "Then five.", "Six."]
    assert xhtml.count('id="e1"') == 1
    assert xhtml.count('class="x"') == 2  # the reopened copy keeps everything else
    assert xhtml.count('name="n1"') == 1
    assert xhtml.count('xml:id="x1"') == 1
    assert xhtml.count('href="#n1"') == 2
