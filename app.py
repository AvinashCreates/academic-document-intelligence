import os
import re
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from sentence_transformers import SentenceTransformer

from ai_detector import detect_ai_writing, load_ai_detector
from preprocessing import (
    clean_academic_text,
    extract_text_from_docx,
    extract_text_from_pdf,
    extract_text_from_txt,
    split_to_paragraphs,
    split_to_sentences,
)
from report_generator import export_report_to_pdf, generate_similarity_report
from similarity import (
    SimilarityResult,
    compare_documents,
    compute_corpus_similarity,
    exact_similarity,
    jaccard_similarity,
    semantic_similarity,
)


st.set_page_config(
    page_title='Academic Document Intelligence',
    layout='wide',
    initial_sidebar_state='expanded',
)

PAGE_OPTIONS = [
    'Upload Document',
    'AI Writing Analysis',
    'Similarity Analysis',
    'Compare Two Papers',
    'Generate Report',
    'Settings',
]

if 'active_page' not in st.session_state:
    st.session_state.active_page = PAGE_OPTIONS[0]


@st.cache_resource
def load_models():
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    ai_detector = load_ai_detector()
    return semantic_model, ai_detector


semantic_model, ai_detector = load_models()


def inject_css() -> None:
    css = '''
    <style>
    .stApp {
      font-family: 'Segoe UI', sans-serif !important;
      background: #f8fafc;
    }

    .simple-shell {
      padding: 1rem 0 2rem;
    }

    .top-header {
      background: linear-gradient(135deg, #0f172a, #1d4ed8);
      color: white;
      padding: 2rem 2rem 1.5rem;
      border-radius: 18px;
      margin-bottom: 1.5rem;
      box-shadow: 0 10px 30px rgba(15, 23, 42, 0.12);
    }

    .simple-title {
      margin: 0;
      font-size: clamp(2rem, 3vw, 3rem);
      line-height: 1.1;
      letter-spacing: -0.04em;
    }

    .simple-subtitle {
      margin: 0.6rem 0 0;
      max-width: 760px;
      color: rgba(255,255,255,0.8);
      font-size: 1rem;
      line-height: 1.7;
    }

    .simple-card {
      background: white;
      border: 1px solid #e2e8f0;
      border-radius: 16px;
      padding: 1.2rem 1.3rem;
      box-shadow: 0 8px 18px rgba(15,23,42,0.04);
    }

    .mini-tag {
      display: inline-block;
      padding: 0.4rem 0.8rem;
      border-radius: 999px;
      background: rgba(255,255,255,0.12);
      border: 1px solid rgba(255,255,255,0.2);
      font-weight: 600;
      font-size: 0.8rem;
      margin-right: 0.5rem;
    }

    .risk-box {
      border-radius: 12px;
      border-left: 6px solid #94a3b8;
      padding: 0.9rem 1rem;
      margin-bottom: 0.8rem;
      background: #f8fafc;
    }

    .metric-box {
      background: white;
      border: 1px solid #e2e8f0;
      border-radius: 14px;
      padding: 1rem;
      min-height: 120px;
      box-shadow: 0 10px 18px rgba(15,23,42,0.03);
    }

    .metric-label {
      font-size: 0.8rem;
      color: #475569;
      margin-bottom: 0.6rem;
    }

    .metric-value {
      font-size: 1.8rem;
      font-weight: 700;
      color: #0f172a;
      margin: 0;
    }
    </style>
    '''
    st.markdown(css, unsafe_allow_html=True)


def save_temp_upload(uploaded_file) -> Path:
    ext = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.getvalue())
        return Path(tmp.name)


def load_uploaded_text(uploaded_file) -> str:
    temp_path = save_temp_upload(uploaded_file)
    try:
        if temp_path.suffix == '.pdf':
            raw = extract_text_from_pdf(str(temp_path))
        elif temp_path.suffix == '.docx':
            raw = extract_text_from_docx(str(temp_path))
        elif temp_path.suffix == '.txt':
            raw = extract_text_from_txt(str(temp_path))
        else:
            raw = ''
    finally:
        temp_path.unlink(missing_ok=True)
    return clean_academic_text(raw)


def load_text_from_path(file_path: str) -> str:
    if file_path.lower().endswith('.pdf'):
        raw = extract_text_from_pdf(file_path)
    elif file_path.lower().endswith('.docx'):
        raw = extract_text_from_docx(file_path)
    elif file_path.lower().endswith('.txt'):
        raw = extract_text_from_txt(file_path)
    else:
        return ''
    return clean_academic_text(raw)


def split_into_paragraphs(text: str, min_words: int = 5, max_paragraphs: Optional[int] = None):
    """Split academic text into meaningful paragraphs without a hard cap on full-document analysis."""
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


def count_total_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ''))


def estimated_section_word_target(text: str, target_sections: int = 30) -> Dict[str, float]:
    total_words = count_total_words(text)
    words_per_section = total_words / max(target_sections, 1)
    return {
        'total_words': total_words,
        'target_sections': target_sections,
        'words_per_section': round(words_per_section, 2),
    }


def get_risk_band(score: float) -> str:
    if score >= 70:
        return 'high'
    if score >= 35:
        return 'medium'
    return 'low'


def get_risk_color(score: float) -> str:
    band = get_risk_band(score)
    if band == 'high':
        return '#fecaca', '#b91c1c'
    if band == 'medium':
        return '#fef3c7', '#b45309'
    return '#dcfce7', '#166534'


def humanize_document_text(text: str, use_hf_model: bool = False) -> str:
    if not text or not text.strip():
        return text

    if use_hf_model:
        try:
            from transformers import pipeline

            generator = pipeline(
                'text2text-generation',
                model='ramsrigouthamg/t5_paraphrase_paws',
                device=-1,
            )
            chunks = [chunk.strip() for chunk in re.split(r'(?<=[.!?])\s+', text) if chunk.strip()]
            rewritten = []
            for chunk in chunks:
                try:
                    result = generator(chunk, max_length=128, num_beams=6, do_sample=True, temperature=0.8)
                    generated = result[0].get('generated_text', chunk).strip()
                    rewritten.append(generated)
                except Exception:
                    rewritten.append(chunk)
            if rewritten:
                return ' '.join(rewritten)
        except Exception:
            pass

    replacements = [
        (r'\bIn conclusion\b', 'To conclude'),
        (r'\bIt is important to note that\b', 'One important point is that'),
        (r'\bThis study demonstrates\b', 'This study shows'),
        (r'\bThe findings suggest\b', 'The results suggest'),
        (r'\bIn addition\b', 'Additionally'),
        (r'\bFurthermore\b', 'Moreover'),
        (r'\bIt can be seen that\b', 'It is evident that'),
        (r'\bIn order to\b', 'To'),
        (r'\bIt is worth mentioning that\b', 'It is helpful to note that'),
        (r'\bThe data indicates\b', 'The data suggests'),
        (r'\bA significant amount of\b', 'A substantial amount of'),
        (r'\bIn the context of\b', 'In the context of the broader discussion around'),
    ]

    sentences = [chunk.strip() for chunk in re.split(r'(?<=[.!?])\s+', text) if chunk.strip()]
    rewritten = []
    for sentence in sentences:
        updated = sentence
        for pattern, replacement in replacements:
            updated = re.sub(pattern, replacement, updated, flags=re.IGNORECASE)
        rewritten.append(updated)
    return ' '.join(rewritten).strip()


def render_full_document_risk_panel(text: str) -> None:
    if not text or not text.strip():
        st.info('Upload a document to analyze the paragraph-level risk map.')
        return

    paragraphs = split_into_paragraphs(text)
    ai_results = analyze_document_paragraphs(text)
    similarity_matches = analyze_document_paragraph_similarity(text, 'corpus') if os.path.isdir('corpus') else []
    similarity_lookup = {item['paragraph_id']: item['overall_similarity'] for item in similarity_matches}
    section_stats = estimated_section_word_target(text, 30)

    st.subheader('Document risk overview')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-box"><div class="metric-label">Total words</div><p class="metric-value">%s</p></div>' % section_stats['total_words'], unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-box"><div class="metric-label">Approx. words / 30 sections</div><p class="metric-value">%.2f</p></div>' % section_stats['words_per_section'], unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-box"><div class="metric-label">Paragraphs analyzed</div><p class="metric-value">%s</p></div>' % len(paragraphs), unsafe_allow_html=True)

    if not paragraphs:
        st.warning('No paragraph-level content was found in this document.')
        return

    st.write('')
    for idx, paragraph in enumerate(paragraphs, start=1):
        ai_score = 0.0
        for item in ai_results:
            if item['paragraph_id'] == idx:
                ai_score = float(item.get('ai_probability', 0.0))
                break
        sim_score = similarity_lookup.get(idx, 0.0)
        total_risk = max(ai_score, sim_score)
        band = get_risk_band(total_risk)
        highlight, text_color = get_risk_color(total_risk)

        st.markdown(
            f"""
            <div class="risk-box" style="background-color:{highlight}; border-left: 6px solid {text_color};">
                <div style="display:flex; justify-content:space-between; gap: 12px; align-items:center; margin-bottom: 8px; flex-wrap:wrap;">
                    <strong>Paragraph {idx}</strong>
                    <span style="font-weight:700; color:{text_color};">{band.upper()} risk · {total_risk:.2f}%</span>
                </div>
                <div>{paragraph}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_humanize_panel(upload_data: Dict[str, any]) -> None:
    if not upload_data.get('file'):
        return

    st.write('## Humanize Full Document')
    st.write('Use this option to rewrite the full document to reduce AI-like phrasing and plagiarism risk. A lightweight fallback rewrite is used automatically if the Hugging Face paraphrase model is not available.')

    use_hf_model = st.checkbox('Try Hugging Face paraphrase model (if available)', value=False)
    if st.button('Humanize document content'):
        with st.spinner('Rewriting the document into a more natural academic style...'):
            rewritten = humanize_document_text(upload_data['text'], use_hf_model=use_hf_model)
        st.success('Document rewritten successfully.')
        st.text_area('Humanized document', rewritten, height=350)
        st.download_button(
            'Download humanized text',
            rewritten,
            file_name='humanized_document.txt',
            mime='text/plain',
        )


def analyze_paragraph_ai(paragraph: str):
    # Call the pipeline and defensively interpret labels.
    raw = ai_detector(paragraph[:2000])[0]
    raw_label = str(raw.get('label', '')).lower()
    score = float(raw.get('score', 0.0))

    # If model returns generic labels like 'LABEL_0', try to map using model config
    mapped_label = raw_label
    try:
        # pipeline exposes the underlying model/config
        cfg = getattr(ai_detector, 'model', None)
        if cfg is not None and hasattr(cfg, 'config'):
            id2label = getattr(cfg.config, 'id2label', None)
            if id2label and raw_label.startswith('label_'):
                idx = int(raw_label.split('_')[-1])
                mapped_label = str(id2label.get(idx, raw_label)).lower()
    except Exception:
        mapped_label = raw_label

    # Interpret mapped label robustly
    if 'real' in mapped_label or 'human' in mapped_label:
        human_prob = score * 100
        ai_prob = (1.0 - score) * 100
    else:
        ai_prob = score * 100
        human_prob = (1.0 - score) * 100

    return {
        'label': raw.get('label', ''),
        'human_probability': round(human_prob, 2),
        'ai_probability': round(ai_prob, 2),
        'raw': raw,
        'mapped_label': mapped_label,
    }


def analyze_document_paragraphs(text: str):
    paragraphs = split_into_paragraphs(text)
    results = []
    if not paragraphs:
        return results
    progress = st.progress(0)
    debug = bool(st.session_state.get('ai_debug', False))
    for i, para in enumerate(paragraphs):
        analysis = analyze_paragraph_ai(para)
        item = {
            'paragraph_id': i + 1,
            'text': para,
            **analysis,
        }
        results.append(item)

        # Debug instrumentation: show raw model output per paragraph
        if debug:
            st.write(f"Paragraph {i+1}")
            st.write('Raw model output:', analysis.get('raw'))
            st.write('Mapped label:', analysis.get('mapped_label'))
            st.write('AI probability:', analysis.get('ai_probability'))
            st.write('Human probability:', analysis.get('human_probability'))

        progress.progress((i + 1) / len(paragraphs))
    return results


def generate_humanization_suggestions(paragraph: str):
    suggestions = []
    if len(paragraph.split()) > 60:
        suggestions.append(
            'Use shorter sentences to improve readability and reduce AI-like structure.'
        )
    if 'significant' in paragraph.lower():
        suggestions.append(
            "Replace vague terms such as 'significant' with specific numerical observations."
        )
    if paragraph.count(',') > 5:
        suggestions.append(
            'Break long comma-separated sentences into smaller statements.'
        )
    suggestions.append(
        'Add your own interpretation, experimental observation, or implementation insight.'
    )
    return suggestions


def get_status_label(prob: float):
    if prob < 20:
        return '🟢 Very likely human-written'
    if prob < 40:
        return '🔵 Mostly human-written'
    if prob < 60:
        return '🟡 Mixed indicators'
    if prob < 80:
        return '🟠 Likely AI-assisted'
    return '🔴 Highly AI-like'


def build_plotly_template(fig):
    theme_base = st.get_option('theme.base') or 'light'
    if theme_base == 'dark':
        fig.update_layout(
            template='plotly_dark',
            font_family='Inter',
            font_color='#f8fafc',
            margin=dict(l=20, r=20, t=36, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        tick_color = '#f8fafc'
    else:
        fig.update_layout(
            template='plotly_white',
            font_family='Inter',
            font_color='#0f172a',
            margin=dict(l=20, r=20, t=36, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        tick_color = '#0f172a'
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False, tickfont=dict(color=tick_color))
    fig.update_yaxes(showgrid=False, zeroline=False, showline=False, tickfont=dict(color=tick_color))
    return fig


def render_sidebar() -> None:
    st.sidebar.title('Document Intelligence')
    st.sidebar.write('Upload, explain, and improve academic content with paragraph-level evidence.')
    st.session_state.active_page = st.sidebar.radio(
        'Navigation',
        PAGE_OPTIONS,
        index=PAGE_OPTIONS.index(st.session_state.active_page),
        label_visibility='collapsed',
    )
    # Debug toggle for AI pipeline diagnostics
    if 'ai_debug' not in st.session_state:
        st.session_state.ai_debug = False
    st.session_state.ai_debug = st.sidebar.checkbox('Enable AI debug logging', value=st.session_state.ai_debug)
    st.sidebar.markdown('---')


def render_header() -> None:
    st.markdown(
        '''
        <div class='dashboard-header'>
          <span class='dashboard-pill'>Explainable AI Writing Analysis</span>
          <h1 class='dashboard-title'>Academic Document Intelligence Platform</h1>
          <p class='dashboard-subtitle'>Upload academic papers to see where AI-style writing appears, inspect semantic similarity evidence, and get humanization suggestions for improvement.</p>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def render_upload_panel(key: str) -> Dict[str, any]:
    st.markdown(
        '''
        <div class='dashboard-upload'>
          <div>
            <p class='upload-title'>Upload your research paper</p>
            <p class='upload-text'>Supported formats: PDF, DOCX, TXT. The system will clean the academic content, remove references, break the document into paragraphs, and analyze each section.</p>
          </div>
          <div>
            <span class='dashboard-pill'>PDF</span>
            <span class='dashboard-pill'>DOCX</span>
            <span class='dashboard-pill'>TXT</span>
          </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        'Select a document to analyze',
        type=['pdf', 'docx', 'txt'],
        key=key,
        label_visibility='hidden',
    )
    if not uploaded_file:
        return {'file': None, 'text': '', 'stats': {}}

    with st.spinner('Extracting and cleaning document text...'):
        text_data = load_uploaded_text(uploaded_file)

    stats = {
        'name': uploaded_file.name,
        'size_mb': f"{uploaded_file.size / 1024 / 1024:.2f} MB",
    }
    st.markdown(
        f"""
        <div class='dashboard-card'>
          <p><strong>Uploaded file:</strong> {stats['name']}</p>
          <p><strong>File size:</strong> {stats['size_mb']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return {'file': uploaded_file, 'text': text_data, 'stats': stats}


def render_document_ai_summary(results: List[Dict[str, any]]) -> None:
    if not results:
        st.warning('No paragraphs were available for AI evidence analysis.')
        return
    avg_ai = sum(r['ai_probability'] for r in results) / len(results)
    avg_human = 100 - avg_ai
    flagged = sum(1 for r in results if r['ai_probability'] >= 40)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('Document AI Probability', f'{avg_ai:.2f}%')
    with col2:
        st.metric('Human Writing Probability', f'{avg_human:.2f}%')
    with col3:
        st.metric('Flagged Paragraphs', flagged)

    if avg_ai < 20:
        st.success('Overall assessment: Very likely human-written')
    elif avg_ai < 40:
        st.info('Overall assessment: Mostly human-written')
    else:
        st.warning('Several sections contain AI-style indicators and may benefit from revision.')


def render_ai_evidence_panel(results: List[Dict[str, any]]) -> None:
    st.write('## AI Writing Evidence Panel')
    for item in results:
        prob = item['ai_probability']
        status = get_status_label(prob)
        with st.expander(f"Paragraph {item['paragraph_id']} — {prob:.2f}% AI probability — {status}"):
            st.write(item['text'])
            st.markdown(
                f"""
                **Human probability:** {item['human_probability']}%  
                **AI probability:** {item['ai_probability']}%  
                **Predicted label:** {item['label']}
                """
            )
            if prob >= 40:
                st.write('### Suggestions to make this section more human-written')
                for suggestion in generate_humanization_suggestions(item['text']):
                    st.write(f'• {suggestion}')

    st.markdown(
        '''
        <div class='dashboard-disclaimer'>
          <strong>Disclaimer:</strong> This analysis highlights the sections that contributed most to the estimated AI-writing probability. It is intended for writing improvement and transparency, not as a definitive authorship determination.
        </div>
        ''',
        unsafe_allow_html=True,
    )


def analyze_document_paragraph_similarity(source_text: str, corpus_dir: str = 'corpus') -> List[Dict[str, any]]:
    """Compare each paragraph against the local corpus and keep the highest matching source per paragraph."""
    paragraphs = split_into_paragraphs(source_text)
    if not paragraphs or not os.path.isdir(corpus_dir):
        return []

    corpus_docs: List[tuple[str, str]] = []
    for root, _, files in os.walk(corpus_dir):
        for filename in sorted(files):
            if not filename.lower().endswith(('.pdf', '.docx', '.txt')):
                continue
            path = os.path.join(root, filename)
            candidate_text = load_text_from_path(path)
            if candidate_text:
                corpus_docs.append((filename, candidate_text))

    if not corpus_docs:
        return []

    paragraph_matches: List[Dict[str, any]] = []
    for idx, paragraph in enumerate(paragraphs, start=1):
        best_match: Dict[str, any] | None = None
        for source_name, reference_text in corpus_docs:
            if not reference_text:
                continue
            exact_score = exact_similarity(paragraph, reference_text)
            jaccard_score = jaccard_similarity(paragraph, reference_text)
            semantic_score = semantic_similarity(paragraph, reference_text, semantic_model)
            overall_score = round((exact_score * 0.4 + jaccard_score * 0.2 + semantic_score * 0.4), 4)
            if best_match is None or overall_score > best_match['overall_similarity']:
                best_match = {
                    'source': source_name,
                    'exact_similarity': exact_score * 100,
                    'jaccard_similarity': jaccard_score * 100,
                    'semantic_similarity': semantic_score * 100,
                    'overall_similarity': overall_score * 100,
                }

        if best_match and best_match['overall_similarity'] >= 5:
            paragraph_matches.append({
                'paragraph_id': idx,
                'text': paragraph,
                **best_match,
            })

    return paragraph_matches


def render_paragraph_similarity_panel(paragraph_results: List[Dict[str, any]], title: str = 'Paragraph plagiarism evidence') -> None:
    if not paragraph_results:
        st.info('No paragraph-level similarities above the threshold were found for this document.')
        return

    st.write(f'## {title}')
    for item in paragraph_results[:30]:
        status = get_status_label(item['overall_similarity'])
        with st.expander(f"Paragraph {item['paragraph_id']} — {item['overall_similarity']:.2f}% match — {item['source']} — {status}"):
            st.write(item['text'])
            st.write(f"**Matched source:** {item['source']}")
            st.write(f"**Overall similarity:** {item['overall_similarity']:.2f}%")
            st.write(f"**Exact similarity:** {item['exact_similarity']:.2f}%")
            st.write(f"**Jaccard similarity:** {item['jaccard_similarity']:.2f}%")
            st.write(f"**Semantic similarity:** {item['semantic_similarity']:.2f}%")


def render_similarity_dashboard(source_text: str, file_name: str) -> None:
    if not os.path.isdir('corpus'):
        st.warning('A local corpus folder is required for similarity evidence. Create a corpus directory with reference documents.')
        return

    with st.spinner('Computing similarity evidence against the local corpus...'):
        results = compute_corpus_similarity(source_text, 'corpus', semantic_model)
        paragraph_matches = analyze_document_paragraph_similarity(source_text, 'corpus')

    if not results:
        st.info('No matching corpus documents were found.')
        return

    st.write('## Semantic Similarity Evidence')
    top_results = results[:5]
    for result in top_results:
        status = get_status_label(result.overall_score * 100)
        with st.expander(f"{result.source} — {result.overall_score * 100:.2f}% overall similarity — {status}"):
            st.write(f"**Exact similarity:** {result.exact_score * 100:.2f}%")
            st.write(f"**Jaccard similarity:** {result.jaccard_score * 100:.2f}%")
            st.write(f"**Semantic similarity:** {result.semantic_score * 100:.2f}%")
            if result.matched_fragments:
                st.write('#### Matched fragments')
                for fragment in result.matched_fragments[:3]:
                    st.markdown(f'- {fragment}')

    similarity_data = pd.DataFrame([
        {
            'source': match.source,
            'overall_similarity': match.overall_score * 100,
            'exact_similarity': match.exact_score * 100,
            'semantic_similarity': match.semantic_score * 100,
        }
        for match in top_results
    ])

    chart = px.bar(
        similarity_data,
        x='overall_similarity',
        y='source',
        orientation='h',
        title='Top Corpus Similarity Matches',
        labels={'overall_similarity': 'Similarity (%)', 'source': 'Document'},
        color='overall_similarity',
        color_continuous_scale=['#22c55e', '#f59e0b', '#dc2626'],
    )
    build_plotly_template(chart)
    st.plotly_chart(chart, use_container_width=True)

    render_paragraph_similarity_panel(paragraph_matches, title='Paragraph plagiarism evidence')


def render_compare_two_papers(file_a, file_b) -> None:
    text_a = load_uploaded_text(file_a)
    text_b = load_uploaded_text(file_b)
    st.write('## Document Comparison Results')

    exact = exact_similarity(text_a, text_b)
    semantic = semantic_similarity(text_a, text_b, semantic_model)
    comparison = compare_documents(text_a, text_b, semantic_model, file_b.name)

    col1, col2, col3 = st.columns(3)
    col1.metric('Exact similarity', f'{exact * 100:.2f}%')
    col2.metric('Semantic similarity', f'{semantic * 100:.2f}%')
    col3.metric('Overall score', f'{comparison.overall_score * 100:.2f}%')

    paragraph_matches: List[Dict[str, any]] = []
    for idx, paragraph in enumerate(split_into_paragraphs(text_a), start=1):
        exact_score = exact_similarity(paragraph, text_b)
        jaccard_score = jaccard_similarity(paragraph, text_b)
        semantic_score = semantic_similarity(paragraph, text_b, semantic_model)
        overall_score = round((exact_score * 0.4 + jaccard_score * 0.2 + semantic_score * 0.4), 4) * 100
        if overall_score >= 5:
            paragraph_matches.append({
                'paragraph_id': idx,
                'text': paragraph,
                'source': file_b.name,
                'exact_similarity': exact_score * 100,
                'jaccard_similarity': jaccard_score * 100,
                'semantic_similarity': semantic_score * 100,
                'overall_similarity': overall_score,
            })

    render_paragraph_similarity_panel(paragraph_matches, title='Paragraph-level similarity between uploaded papers')

    st.write('### Top Matched Fragments')
    if comparison.matched_fragments:
        for fragment in comparison.matched_fragments[:5]:
            st.markdown(f'- {fragment}')
    else:
        st.info('No strong matched fragments were detected.')

    st.write('### Comparison details')
    st.write(
        f"Source documents: **{file_a.name}** and **{file_b.name}** — this comparison reports exact, Jaccard, and semantic similarity evidence."
    )


def render_report_page(upload_data: Dict[str, any]) -> None:
    if not upload_data['file']:
        return

    st.write('## Transparent Document Report')
    with st.spinner('Generating transparent document report...'):
        ai_results = analyze_document_paragraphs(upload_data['text'])
        report = generate_similarity_report([], {
            'ai_probability': sum(r['ai_probability'] for r in ai_results) / len(ai_results) if ai_results else 0,
            'human_probability': 100 - (sum(r['ai_probability'] for r in ai_results) / len(ai_results) if ai_results else 0),
            'predicted_label': 'AI evidence report',
            'model_confidence': round(sum(r['ai_probability'] for r in ai_results) / len(ai_results) if ai_results else 0, 2),
            'confidence_band': get_status_label(sum(r['ai_probability'] for r in ai_results) / len(ai_results) if ai_results else 0),
            'top_matches': [],
            'matched_sources': 0,
        })

    st.markdown('### Report snapshot')
    st.write(f"**Document:** {upload_data['stats'].get('name', 'Uploaded document')}")
    if ai_results:
        render_document_ai_summary(ai_results)
    download_name = f"report_{upload_data['stats'].get('name', 'document')}.pdf"
    pdf_path = export_report_to_pdf(report, download_name)
    with open(pdf_path, 'rb') as file:
        st.download_button('Download full PDF report', file, file_name=download_name, mime='application/pdf')


def render_footer() -> None:
    st.markdown(
        '''
        <div class='dashboard-footer'>
          Built for academic writing transparency, project demonstration, and research-quality review workflows.
        </div>
        ''',
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_css()
    st.markdown('<div class="simple-shell">', unsafe_allow_html=True)
    st.markdown(
        '''
        <div class="top-header">
            <div style="display:flex; flex-wrap:wrap; gap:0.5rem; margin-bottom: 0.8rem;">
                <span class="mini-tag">AI Review</span>
                <span class="mini-tag">Plagiarism Check</span>
                <span class="mini-tag">Humanize</span>
            </div>
            <h1 class="simple-title">Academic Document Intelligence</h1>
            <p class="simple-subtitle">Upload a paper, review paragraph-level risk, compare similarity evidence, and rewrite the whole document into a more natural academic style.</p>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    uploaded = render_upload_panel('upload_main')
    if not uploaded['file']:
        st.info('Upload a PDF, DOCX, or TXT file to start the analysis.')
        st.markdown('</div>', unsafe_allow_html=True)
        return

    text = uploaded['text']
    paragraphs = split_into_paragraphs(text)

    if not paragraphs:
        st.warning('No readable paragraphs were found in the uploaded document.')
        st.markdown('</div>', unsafe_allow_html=True)
        return

    st.markdown('### Document summary')
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric('Word count', count_total_words(text))
    with c2:
        st.metric('Paragraphs', len(paragraphs))
    with c3:
        st.metric('Document size', uploaded['stats'].get('size_mb', '0 MB'))

    st.markdown('---')

    tab_ai, tab_similarity, tab_humanize = st.tabs(['AI Writing Analysis', 'Similarity Evidence', 'Humanize Document'])

    with tab_ai:
        results = analyze_document_paragraphs(text)
        if results:
            render_document_ai_summary(results)
        render_full_document_risk_panel(text)
        render_ai_evidence_panel(results if results else [])

    with tab_similarity:
        render_similarity_dashboard(text, uploaded['stats']['name'])

    with tab_humanize:
        render_humanize_panel(uploaded)

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()
