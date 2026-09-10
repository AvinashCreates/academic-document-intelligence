from typing import Dict, List

import torch
from transformers import pipeline


def load_ai_detector():
    """Load the AI text detection model once and cache it in Streamlit."""
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        'text-classification',
        model='openai-community/roberta-base-openai-detector',
        device=device,
        return_all_scores=False,
    )


def normalize_detector_label(raw_label: str, id2label: Dict[int, str] = None) -> str:
    """Normalize model labels to a human-readable predicted label.

    If an id2label mapping is provided, use it to map LABEL_0/LABEL_1 to semantics.
    """
    label = raw_label.strip().lower()
    # If a direct mapping exists in id2label (e.g., {"0": "REAL"}), prefer that
    if id2label and label.startswith('label_'):
        try:
            idx = int(label.split('_')[-1])
            mapped = str(id2label.get(idx, '')).strip().lower()
            if mapped:
                label = mapped
        except Exception:
            pass

    if label in {'real', 'human', 'human-written', 'human_written', 'authentic'}:
        return 'Real'
    if label in {'fake', 'ai', 'generated', 'ai-generated', 'machine', 'artificial'}:
        return 'AI-generated'
    # Conservative default: we cannot assume AI if unsure
    return 'Real'


def chunk_text(text: str, max_chars: int = 512, overlap: int = 128) -> List[str]:
    """Split text into overlapping chunks for more robust document-level analysis."""
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + max_chars, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_length:
            break
        start = max(end - overlap, start + 1)
    return chunks


def interpret_model_result(label: str, confidence: float, id2label: Dict[int, str] = None) -> Dict[str, float]:
    """Convert model predictions into AI and human probabilities.

    Accepts an optional id2label mapping to correctly interpret LABEL_X outputs.
    """
    predicted_label = normalize_detector_label(label, id2label=id2label)
    if predicted_label == 'Real':
        human_probability = confidence
        ai_probability = 1.0 - confidence
    else:
        ai_probability = confidence
        human_probability = 1.0 - confidence
    return {
        'predicted_label': predicted_label,
        'model_confidence': confidence,
        'ai_probability': ai_probability,
        'human_probability': human_probability,
    }


def confidence_band(ai_probability: float) -> str:
    """Provide a human-readable confidence band for AI probability."""
    if ai_probability < 0.20:
        return 'Very likely human-written'
    if ai_probability < 0.40:
        return 'Mostly human-written'
    if ai_probability < 0.60:
        return 'Mixed / uncertain'
    if ai_probability < 0.80:
        return 'Likely AI-assisted'
    return 'Highly likely AI-generated'


def detect_ai_writing(text: str, ai_detector) -> Dict[str, float]:
    """Analyze a document in chunks and return aggregated AI/human probabilities."""
    cleaned_text = text.strip()
    if not cleaned_text:
        return {
            'label': 'Unknown',
            'model_confidence': 0.0,
            'ai_probability': 0.0,
            'human_probability': 0.0,
            'confidence_band': 'No text to analyze',
            'chunk_count': 0,
            'sample_count': 0,
        }

    chunks = chunk_text(cleaned_text, max_chars=512, overlap=128)
    chunk_results: List[Dict[str, float]] = []

    # If the pipeline exposes the model config, try to obtain id2label mapping
    id2label = None
    try:
        model_obj = getattr(ai_detector, 'model', None)
        if model_obj is not None and hasattr(model_obj, 'config'):
            id2label = getattr(model_obj.config, 'id2label', None)
    except Exception:
        id2label = None

    for chunk in chunks:
        result = ai_detector(chunk)[0]
        structured = interpret_model_result(result['label'], float(result['score']), id2label=id2label)
        chunk_results.append(structured)

    if not chunk_results:
        return {
            'label': 'Unknown',
            'model_confidence': 0.0,
            'ai_probability': 0.0,
            'human_probability': 0.0,
            'confidence_band': 'No chunks generated',
            'chunk_count': 0,
            'sample_count': 0,
        }

    avg_ai = sum(chunk['ai_probability'] for chunk in chunk_results) / len(chunk_results)
    avg_human = sum(chunk['human_probability'] for chunk in chunk_results) / len(chunk_results)

    label_votes = {'Real': 0, 'AI-generated': 0}
    label_confidences = {'Real': [], 'AI-generated': []}
    for chunk in chunk_results:
        label_votes[chunk['predicted_label']] += 1
        label_confidences[chunk['predicted_label']].append(chunk['model_confidence'])

    final_label = 'Real' if label_votes['Real'] >= label_votes['AI-generated'] else 'AI-generated'
    final_confidence_values = label_confidences[final_label] or [1.0 - avg_ai if final_label == 'Real' else avg_ai]
    final_confidence = sum(final_confidence_values) / len(final_confidence_values)

    return {
        'label': final_label,
        'model_confidence': round(final_confidence * 100, 2),
        'ai_probability': round(avg_ai * 100, 2),
        'human_probability': round(avg_human * 100, 2),
        'confidence_band': confidence_band(avg_ai),
        'chunk_count': len(chunks),
        'sample_count': len(chunk_results),
    }
