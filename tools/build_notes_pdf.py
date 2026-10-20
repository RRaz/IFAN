"""Typeset NOTES_FOR_PHOTOGRAPHERS.md into assets/notes/notes-for-photographers.pdf.

The notes exist twice: as markdown, and as a typeset PDF that the markdown points at as
the original. Keeping the two in step by hand is how the PDF came to carry no licence at
all: the CC grant was added to the markdown afterwards and the PDF was never rebuilt. So
the PDF is built from the markdown, and the licence is printed twice — a one-line grant
under the title, the full wording at the end — rather than depending on a reader reaching
the bottom.

Why a browser does the printing: LibreOffice's headless HTML import silently drops every
`<img>`, so a rebuild through it produced three pages of text and no photographs. Edge and
Chrome both honour the CSS below and embed the images.

    python tools/build_notes_pdf.py

Needs Edge or Chrome and nothing else. The markdown is the only source; edit it, run this,
commit both.
"""

import html
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, 'NOTES_FOR_PHOTOGRAPHERS.md')
TARGET = os.path.join(ROOT, 'assets', 'notes', 'notes-for-photographers.pdf')

BROWSERS = (
    r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
    r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
    r'C:\Program Files\Google\Chrome\Application\chrome.exe',
    r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
    'google-chrome',
    'chromium',
)

LICENCE_URL = 'https://creativecommons.org/licenses/by-nc-nd/4.0/'

STYLE = """
@page { size: A4; margin: 20mm 22mm 20mm 22mm; }
body { font-family: Georgia, 'Times New Roman', serif; font-size: 10.5pt;
       line-height: 1.5; color: #1b1b1b; }
h1 { font-size: 20pt; line-height: 1.2; margin: 0 0 4mm 0; color: #111; }
h2 { font-size: 13pt; margin: 9mm 0 3mm 0; color: #111; page-break-after: avoid; }
p { margin: 0 0 3.2mm 0; text-align: justify; }
.byline { font-style: italic; color: #555; font-size: 9.5pt; margin-bottom: 1mm; }
.grant { font-size: 9pt; color: #555; border-top: 0.4pt solid #bbb;
         border-bottom: 0.4pt solid #bbb; padding: 2.2mm 0; margin: 0 0 8mm 0; }
.caption { font-style: italic; color: #444; font-size: 8.8pt; text-align: left;
           margin: -1mm 0 4mm 0; }
figure { margin: 4mm 0 1mm 0; page-break-inside: avoid; }
img { width: 100%; }
hr { border: 0; border-top: 0.4pt solid #ccc; margin: 8mm 0; }
a { color: #1a4f8a; text-decoration: none; }
"""


def inline(text):
    """Markdown inline: links, bold, italic, in that order."""
    text = html.escape(text, quote=False)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)',
                  lambda m: '<a href="{}">{}</a>'.format(m.group(2), m.group(1)), text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<em>\1</em>', text)
    return text.replace('---', '\u2014')


def build_html(markdown, work):
    """The markdown this file is written in, as the HTML the browser prints."""
    grant = ('<p class="grant">&copy; 2026 Ryan Raz &nbsp;&middot;&nbsp; text AGPL-3.0 '
             '&nbsp;&middot;&nbsp; photographs '
             '<a href="{}">CC BY-NC-ND 4.0</a></p>'.format(LICENCE_URL))

    body = []
    paragraph = []
    skipping = False

    def flush():
        if paragraph:
            body.append('<p>{}</p>'.format(inline(' '.join(paragraph))))
            del paragraph[:]

    for line in markdown.splitlines():
        stripped = line.strip()

        if not stripped:
            flush()
            skipping = False
            continue

        # The markdown's opening note points at this very PDF; in the PDF it becomes a
        # plain byline instead, and the rest of that paragraph is dropped with it.
        if stripped.startswith('*Written by the author'):
            flush()
            skipping = True
            body.append('<p class="byline">Written by the author of this fork, from his '
                        'own photographs, October 2026.</p>')
            body.append(grant)
            continue

        if skipping:
            continue

        if stripped.startswith('# '):
            flush()
            body.append('<h1>{}</h1>'.format(inline(stripped[2:])))
            continue

        if stripped.startswith('## '):
            flush()
            body.append('<h2>{}</h2>'.format(inline(stripped[3:])))
            continue

        if stripped == '---':
            flush()
            body.append('<hr>')
            continue

        image = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', stripped)
        if image:
            flush()
            alt, path = image.groups()
            path = os.path.join(ROOT, path.replace('/', os.sep))
            # A copy beside the HTML, referenced by name: a browser will load it, and the
            # relative reference keeps the page portable if it is opened by hand.
            shutil.copyfile(path, os.path.join(work, os.path.basename(path)))
            body.append('<figure><img src="{}" alt="{}"></figure>'
                        .format(os.path.basename(path), html.escape(alt, quote=True)))
            continue

        # A line wrapped in single asterisks is a figure caption.
        if stripped.startswith('*') and stripped.endswith('*') and len(stripped) > 2:
            flush()
            body.append('<p class="caption">{}</p>'.format(inline(stripped)))
            continue

        paragraph.append(stripped)

    flush()

    # The markdown's own Provenance and Licence paragraphs stay in the body: they are the
    # canonical wording. The grant above is the form that is seen without scrolling.
    return ('<!DOCTYPE html>\n<html><head><meta charset="utf-8">'
            '<title>Notes for photographers</title><style>{}</style></head><body>'
            '{}</body></html>'.format(STYLE, '\n'.join(body)))


def find_browser():
    """First Chromium-family browser on this machine, or None."""
    for candidate in BROWSERS:
        if os.path.isabs(candidate):
            if os.path.isfile(candidate):
                return candidate
        else:
            found = shutil.which(candidate)
            if found:
                return found
    return None


def main():
    browser = find_browser()
    if not browser:
        sys.exit('No Edge or Chrome found; one of them prints this PDF.')

    markdown = open(SOURCE, encoding='utf-8').read()
    work = tempfile.mkdtemp(prefix='notes-pdf-')
    page = os.path.join(work, 'notes.html')
    built = os.path.join(work, 'notes.pdf')

    with open(page, 'w', encoding='utf-8') as handle:
        handle.write(build_html(markdown, work))

    result = subprocess.run([browser, '--headless=new', '--disable-gpu',
                             '--no-pdf-header-footer', '--virtual-time-budget=8000',
                             '--print-to-pdf=' + built,
                             'file:///' + page.replace(os.sep, '/')],
                            capture_output=True, text=True)
    if not os.path.isfile(built):
        print(result.stdout)
        print(result.stderr)
        sys.exit('the browser did not produce a PDF')

    data = open(built, 'rb').read()
    open(TARGET, 'wb').write(data)
    shutil.rmtree(work, ignore_errors=True)
    print('wrote {} ({:.0f} KB)'.format(os.path.relpath(TARGET, ROOT), len(data) / 1024.0))


if __name__ == '__main__':
    main()
