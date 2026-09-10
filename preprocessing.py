import re
from typing import List

import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt', quiet=True)


def extract_text_from_pdf(path: str) -> str:
    """Extract text from a PDF file path."""
    from PyPDF2 import PdfReader

    reader = PdfReader(path)
    text_parts: List[str] = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    return '\n'.join(text_parts)


def extract_text_from_docx(path: str) -> str:
    """Extract text from a DOCX file path."""
    import docx2txt

    return docx2txt.process(path) or ''


def extract_text_from_txt(path: str) -> str:
    """Extract text from a plain text file path."""
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def remove_references(text: str) -> str:
    """Remove common bibliography sections from academic text."""
    patterns = [
        r'(?i)\nreferences\n.*$',
        r'(?i)\nbibliography\n.*$',
        r'(?i)\nworks cited\n.*$',
        r'(?i)\nreference list\n.*$',
        r'(?i)\nnotes\n.*$',
    ]
    for pattern in patterns:
        text = re.sub(pattern, '\n', text, flags=re.DOTALL)
    return text


def remove_headers_and_footers(text: str) -> str:
    """Strip repeating headers and footer lines from text."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 10:
        return text

    header = lines[0]
    footer = lines[-1]
    cleaned = [line for line in lines if line not in (header, footer)]
    return '\n'.join(cleaned)


def remove_tables_and_captions(text: str) -> str:
    """Remove table and figure captions from the text."""
    text = re.sub(r'(?i)table\s*\d+.*?\n', ' ', text)
    text = re.sub(r'(?i)figure\s*\d+.*?\n', ' ', text)
    return text


def clean_academic_text(text: str) -> str:
    """Clean academic text by removing noisy academic sections and repeated elements."""
    # Normalize newlines first and preserve paragraph separators.
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Collapse multiple blank lines to at most one paragraph separator
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # Collapse repeated spaces and tabs but keep newlines
    text = re.sub(r'[ \t]+', ' ', text)

    # Strip whitespace at line ends
    lines = [line.strip() for line in text.splitlines()]
    cleaned = '\n'.join(lines)

    cleaned = remove_headers_and_footers(cleaned)
    cleaned = remove_tables_and_captions(cleaned)
    cleaned = remove_references(cleaned)
    cleaned = re.sub(r'\[[0-9]+\]', ' ', cleaned)
    cleaned = re.sub(r'\([^)]*?\bTable\b[^)]*?\)', ' ', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\([^)]*?\bFigure\b[^)]*?\)', ' ', cleaned, flags=re.IGNORECASE)

    # Normalize any repeated whitespace but preserve paragraph breaks
    cleaned = re.sub(r"[ \t]+", ' ', cleaned)
    cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
    return cleaned.strip()


def split_to_sentences(text: str) -> List[str]:
    """Split clean academic text into sentences."""
    return [sentence.strip() for sentence in sent_tokenize(text) if sentence.strip()]


def split_to_paragraphs(text: str, min_words: int = 5, max_paragraphs: int | None = None) -> List[str]:
    """Split clean academic text into meaningful paragraphs without a hard cap on full-document analysis."""
    paragraphs: List[str] = []
    for raw_paragraph in re.split(r"\n\s*\n+|\r\n\s*\r\n+", text):
        paragraph = re.sub(r"\s+", ' ', raw_paragraph).strip()
        if not paragraph:
            continue

        word_count = len(paragraph.split())
        if word_count < min_words:
            if word_count < 3 and not any(ch in paragraph for ch in '.!?'):
                continue

        paragraphs.append(paragraph)
        if max_paragraphs is not None and len(paragraphs) >= max_paragraphs:
            break
    return paragraphs
