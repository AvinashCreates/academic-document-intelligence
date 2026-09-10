import os
from dataclasses import dataclass
from typing import List

from datasketch import MinHash
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import (
    clean_academic_text,
    extract_text_from_docx,
    extract_text_from_pdf,
    extract_text_from_txt,
)


@dataclass
class SimilarityResult:
    source: str
    exact_score: float
    jaccard_score: float
    semantic_score: float
    overall_score: float
    matched_fragments: List[str]


def tokenize_text(text: str) -> List[str]:
    """Split text into tokens for similarity analysis."""
    return [token for token in text.lower().split() if token.isalnum()]


def exact_similarity(text1: str, text2: str) -> float:
    """Compute exact similarity using TF-IDF cosine similarity."""
    vectorizer = TfidfVectorizer(ngram_range=(3, 5), stop_words='english')
    corpus = [text1, text2]
    tfidf_matrix = vectorizer.fit_transform(corpus)
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return float(score)


def jaccard_similarity(text1: str, text2: str, num_perm: int = 128) -> float:
    """Compute Jaccard similarity using MinHash on token shingles."""
    tokens1 = tokenize_text(text1)
    tokens2 = tokenize_text(text2)
    minhash1 = MinHash(num_perm=num_perm)
    minhash2 = MinHash(num_perm=num_perm)
    for token in tokens1:
        minhash1.update(token.encode('utf8'))
    for token in tokens2:
        minhash2.update(token.encode('utf8'))
    return float(minhash1.jaccard(minhash2))


def semantic_similarity(text1: str, text2: str, model: SentenceTransformer) -> float:
    """Compute semantic similarity using Sentence-BERT."""
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1]).item()
    return float(score)


def compare_documents(
    source_text: str,
    candidate_text: str,
    model: SentenceTransformer,
    source_name: str,
) -> SimilarityResult:
    """Compare two documents and return a combined similarity result."""
    exact_score = exact_similarity(source_text, candidate_text)
    jaccard_score = jaccard_similarity(source_text, candidate_text)
    semantic_score = semantic_similarity(source_text, candidate_text, model)
    overall_score = round((exact_score * 0.4 + jaccard_score * 0.2 + semantic_score * 0.4), 4)

    matched_fragments: List[str] = []
    candidate_sentences = [sentence.strip() for sentence in candidate_text.split('.') if len(sentence.split()) > 8]
    for sentence in candidate_sentences:
        ratio = fuzz.partial_ratio(sentence, source_text)
        if ratio > 75:
            matched_fragments.append(sentence)

    return SimilarityResult(
        source=source_name,
        exact_score=exact_score,
        jaccard_score=jaccard_score,
        semantic_score=semantic_score,
        overall_score=overall_score,
        matched_fragments=matched_fragments,
    )


def _load_corpus_text(file_path: str) -> str:
    """Load text from a local corpus file and clean academic content."""
    if file_path.lower().endswith('.pdf'):
        raw_text = extract_text_from_pdf(file_path)
    elif file_path.lower().endswith('.docx'):
        raw_text = extract_text_from_docx(file_path)
    elif file_path.lower().endswith('.txt'):
        raw_text = extract_text_from_txt(file_path)
    else:
        return ''
    return clean_academic_text(raw_text)


def compute_corpus_similarity(
    source_text: str,
    corpus_dir: str,
    model: SentenceTransformer,
) -> List[SimilarityResult]:
    """Compare source text against all documents in a local corpus directory."""
    if not os.path.isdir(corpus_dir):
        return []

    results: List[SimilarityResult] = []
    for root, _, files in os.walk(corpus_dir):
        for filename in files:
            if filename.lower().endswith(('.pdf', '.docx', '.txt')):
                file_path = os.path.join(root, filename)
                candidate_text = _load_corpus_text(file_path)
                if not candidate_text:
                    continue
                result = compare_documents(source_text, candidate_text, model, filename)
                results.append(result)
    return sorted(results, key=lambda r: r.overall_score, reverse=True)
