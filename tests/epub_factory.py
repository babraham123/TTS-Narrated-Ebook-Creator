"""Builds small, epubcheck-valid EPUB 3 books for the test suite."""

import zipfile

CHAPTERS = [
    (
        "Chapter One",
        [
            "It was a bright cold day in April. The clocks were striking thirteen.",
            'Tom &amp; Jerry <em class="loud">ran fast. Then</em> they stopped &lt;suddenly&gt;.',
            'She said <span epub:type="pagebreak" id="pg1" title="1"/>nothing at all.',
        ],
    ),
    (
        "Chapter Two",
        [
            "A short one.",
            "Here is a longer paragraph with <a href=\"ch1.xhtml\">a link</a> inside it. "
            "And a second sentence follows the first.",
        ],
    ),
]


def _chapter_xhtml(title, paragraphs):
    body = "\n".join(f"<p>{p}</p>" for p in paragraphs)
    return f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="en" lang="en">
<head>
<title>{title}</title>
<link rel="stylesheet" type="text/css" href="../Styles/style.css"/>
</head>
<body>
<section epub:type="chapter">
<h1>{title}</h1>
{body}
</section>
</body>
</html>
"""


NAV = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="en" lang="en">
<head><title>Contents</title></head>
<body>
<nav epub:type="toc" id="toc">
<h1>Contents</h1>
<ol>
{items}
</ol>
</nav>
</body>
</html>
"""

OPF = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid" xml:lang="en">
<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
<dc:identifier id="bookid">urn:uuid:12345678-1234-4234-8234-123456789abc</dc:identifier>
<dc:title>Test Book</dc:title>
<dc:language>en</dc:language>
<meta property="dcterms:modified">2024-01-01T00:00:00Z</meta>
</metadata>
<manifest>
<item id="nav" href="Text/nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
<item id="css" href="Styles/style.css" media-type="text/css"/>
{items}
</manifest>
<spine>
{itemrefs}
</spine>
</package>
"""

CONTAINER = """<?xml version="1.0" encoding="utf-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
<rootfiles>
<rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
</rootfiles>
</container>
"""


def make_epub(path, chapters=CHAPTERS):
    names = [f"ch{i + 1}" for i in range(len(chapters))]
    items = "\n".join(
        f'<item id="{n}" href="Text/{n}.xhtml" media-type="application/xhtml+xml"/>' for n in names
    )
    itemrefs = "\n".join(f'<itemref idref="{n}"/>' for n in names)
    nav_items = "\n".join(
        f'<li><a href="{n}.xhtml">{title}</a></li>' for n, (title, _) in zip(names, chapters)
    )
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        z.writestr("META-INF/container.xml", CONTAINER)
        z.writestr("OEBPS/content.opf", OPF.format(items=items, itemrefs=itemrefs))
        z.writestr("OEBPS/Text/nav.xhtml", NAV.format(items=nav_items))
        z.writestr("OEBPS/Styles/style.css", "p { margin: 0; }\n")
        for n, (title, paragraphs) in zip(names, chapters):
            z.writestr(f"OEBPS/Text/{n}.xhtml", _chapter_xhtml(title, paragraphs))
    return path
