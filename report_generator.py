from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

from similarity import SimilarityResult


def generate_similarity_report(results: List[SimilarityResult], ai_result: Dict[str, float]) -> Dict[str, object]:
    """Generate a structured similarity report dictionary."""
    matched_sources = [res for res in results if res.overall_score > 0.1]
    top_matches = matched_sources[:10]

    # Accept both 'label' and 'predicted_label' keys for compatibility with different callers
    label = ai_result.get('label') or ai_result.get('predicted_label') or 'Unknown'
    report = {
        'overall_similarity': round(sum(r.overall_score for r in matched_sources) / max(len(matched_sources), 1) * 100, 2),
        'ai_probability': ai_result.get('ai_probability', 0.0),
        'human_probability': ai_result.get('human_probability', 100.0 - ai_result.get('ai_probability', 0.0)),
        'predicted_label': label,
        'model_confidence': ai_result.get('model_confidence', 0.0),
        'confidence_band': ai_result.get('confidence_band', 'Unknown'),
        'chunk_count': ai_result.get('chunk_count', 0),
        'matched_sources': len(matched_sources),
        'top_matches': [asdict(match) for match in top_matches],
        'sources': [asdict(result) for result in results],
    }
    return report


def export_report_to_pdf(report: Dict[str, object], output_path: str) -> str:
    """Export the similarity report structure to a PDF file."""
    path = Path(output_path)
    doc = SimpleDocTemplate(str(path), pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph('Academic Similarity Report', styles['Title']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph('This tool provides academic similarity estimation and AI-writing probability analysis. It is not affiliated with or equivalent to Turnitin or iThenticate.', styles['Normal']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"Overall Similarity: {report['overall_similarity']}%", styles['Heading2']))
    elements.append(Paragraph(f"AI Writing Probability: {report['ai_probability']}%", styles['Heading2']))
    elements.append(Paragraph(f"Matched Sources: {report['matched_sources']}", styles['Heading2']))
    elements.append(Spacer(1, 12))

    table_data = [['Source', 'Exact', 'Jaccard', 'Semantic', 'Overall']]
    for result in report['top_matches']:
        table_data.append([
            result['source'],
            f"{result['exact_score']:.2f}",
            f"{result['jaccard_score']:.2f}",
            f"{result['semantic_score']:.2f}",
            f"{result['overall_score']:.2f}",
        ])

    table = Table(table_data, colWidths=[180, 70, 70, 70, 70])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))

    for match in report['top_matches']:
        elements.append(Paragraph(f"Source: {match['source']}", styles['Heading3']))
        for fragment in match['matched_fragments'][:3]:
            elements.append(Paragraph(fragment, styles['Normal']))
            elements.append(Spacer(1, 6))
        elements.append(Spacer(1, 12))

    doc.build(elements)
    return str(path)
