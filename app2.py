import streamlit as st
import pandas as pd
import nltk
import tempfile
from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup
import requests
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import io
import os
import docx2txt
from PyPDF2 import PdfReader
import plotly.express as px

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

@st.cache_resource
def load_models():
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

    ai_detector = pipeline(
        "text-classification",
        model="openai-community/roberta-base-openai-detector"
    )

    return semantic_model, ai_detector

semantic_model, ai_detector = load_models()

def get_sentences(text):
    return sent_tokenize(text)

def get_url(sentence):
    base_url = 'https://www.google.com/search?q='
    query = sentence
    query = query.replace(' ', '+')
    url = base_url + query
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'
    }
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code != 200:
            return None
    except requests.exceptions.RequestException:
        return None

    soup = BeautifulSoup(res.text, 'html.parser')
    divs = soup.find_all('div', class_='yuRUbf')
    for div in divs:
        a = div.find('a')
        if a and a.get('href'):
            link = a['href']
            if 'youtube' not in link:
                return link
    return None

def read_text_file(file):
    data = file.getvalue() if hasattr(file, 'getvalue') else file.read()
    if isinstance(data, bytes):
        return data.decode('utf-8', errors='ignore')
    return str(data)

def read_docx_file(file):
    file_bytes = file.getvalue() if hasattr(file, 'getvalue') else file.read()
    if isinstance(file_bytes, str):
        file_bytes = file_bytes.encode('utf-8', errors='ignore')
    with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        text = docx2txt.process(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return text

def read_pdf_file(file):
    file_bytes = file.getvalue() if hasattr(file, 'getvalue') else file.read()
    if isinstance(file_bytes, str):
        file_bytes = file_bytes.encode('utf-8', errors='ignore')
    with io.BytesIO(file_bytes) as bytes_io:
        pdf_reader = PdfReader(bytes_io)
        text = ''
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_from_file(uploaded_file):
    content = ""
    if uploaded_file is not None:
        file_type = uploaded_file.type or ''
        file_name = uploaded_file.name.lower()
        if file_type == "text/plain" or file_name.endswith('.txt'):
            content = read_text_file(uploaded_file)
        elif file_type == "application/pdf" or file_name.endswith('.pdf'):
            content = read_pdf_file(uploaded_file)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or file_name.endswith('.docx'):
            content = read_docx_file(uploaded_file)
    return content

def get_text(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return ''
    except requests.exceptions.RequestException:
        return ''
    soup = BeautifulSoup(response.text, 'html.parser')
    text = ' '.join([p.get_text() for p in soup.find_all('p')])
    return text

def get_similarity(text1, text2):
    """Semantic similarity using Sentence-BERT"""
    emb1 = semantic_model.encode(text1, convert_to_tensor=True)
    emb2 = semantic_model.encode(text2, convert_to_tensor=True)

    similarity = util.cos_sim(emb1, emb2).item()
    return float(similarity)

def get_similarity_list(texts, filenames=None):
    similarity_list = []
    if filenames is None:
        filenames = [f"File {i+1}" for i in range(len(texts))]
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            similarity = get_similarity(texts[i], texts[j])
            similarity_list.append((filenames[i], filenames[j], similarity))
    return similarity_list
def get_similarity_list2(text, url_list):
    similarity_list = []
    for url in url_list:
        text2 = get_text(url)
        similarity = get_similarity(text, text2)
        similarity_list.append(similarity)
    return similarity_list

def get_sentence_similarity_list(sentences):
    similarity_list = []
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            similarity = get_similarity(sentences[i], sentences[j])
            similarity_list.append((f"Sentence {i+1}", f"Sentence {j+1}", similarity, sentences[i], sentences[j]))
    return similarity_list

def detect_ai_similarity(text):
    """Detect probability of AI-generated text"""
    sample = text[:512]

    result = ai_detector(sample)[0]

    return {
        "label": result['label'],
        "confidence": round(result['score'] * 100, 2)
    }

def plot_scatter(df):
    fig = px.scatter(df, x='File 1', y='File 2', color='Similarity', title='Similarity Scatter Plot')
    st.plotly_chart(fig, use_container_width=True)

def plot_line(df):
    fig = px.line(df, x='File 1', y='File 2', color='Similarity', title='Similarity Line Chart')
    st.plotly_chart(fig, use_container_width=True)

def plot_bar(df):
    fig = px.bar(df, x='File 1', y='Similarity', color='File 2', title='Similarity Bar Chart')
    st.plotly_chart(fig, use_container_width=True)

def plot_pie(df):
    fig = px.pie(df, values='Similarity', names='File 1', title='Similarity Pie Chart')
    st.plotly_chart(fig, use_container_width=True)

def plot_box(df):
    fig = px.box(df, x='File 1', y='Similarity', title='Similarity Box Plot')
    st.plotly_chart(fig, use_container_width=True)

def plot_histogram(df):
    fig = px.histogram(df, x='Similarity', title='Similarity Histogram')
    st.plotly_chart(fig, use_container_width=True)

def plot_3d_scatter(df):
    fig = px.scatter_3d(df, x='File 1', y='File 2', z='Similarity', color='Similarity',
                        title='Similarity 3D Scatter Plot')
    st.plotly_chart(fig, use_container_width=True)

def plot_violin(df):
    fig = px.violin(df, y='Similarity', x='File 1', title='Similarity Violin Plot')
    st.plotly_chart(fig, use_container_width=True)



st.set_page_config(page_title='Plagiarism Detection')
st.title('Plagiarism Detector')

st.write("""
### Enter the text or upload a file to check for plagiarism or find similarities between files
""")
option = st.radio(
    "Select input option:",
    ('Enter text', 'Upload file', 'Find similarities between files')
)

if option == 'Enter text':
    text = st.text_area("Enter text here", height=200)
    uploaded_files = []
elif option == 'Upload file':
    uploaded_file = st.file_uploader("Upload file (.docx, .pdf, .txt)", type=["docx", "pdf", "txt"])
    if uploaded_file is not None:
        try:
            text = get_text_from_file(uploaded_file)
            if not text.strip():
                st.warning("Uploaded file contained no readable text. Please try a different file.")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            text = ""
        uploaded_files = [uploaded_file]
        st.write(f"Uploaded file: {uploaded_file.name}")
        st.write(f"Extracted text length: {len(text)} characters")
    else:
        text = ""
        uploaded_files = []
else:
    uploaded_files = st.file_uploader("Upload multiple files (.docx, .pdf, .txt)", type=["docx", "pdf", "txt"], accept_multiple_files=True)
    texts = []
    filenames = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file is not None:
                try:
                    file_text = get_text_from_file(uploaded_file)
                    if not file_text.strip():
                        st.warning(f"Uploaded {uploaded_file.name} contained no readable text.")
                except Exception as e:
                    st.error(f"Error reading file {uploaded_file.name}: {e}")
                    file_text = ""
                texts.append(file_text)
                filenames.append(uploaded_file.name)
        st.write("Uploaded files:", [f.name for f in uploaded_files])
        st.write("Extracted text lengths:", [len(t) for t in texts])
    text = " ".join(texts)

col1, col2 = st.columns(2)

with col1:
    plagiarism_btn = st.button('Check Plagiarism')

with col2:
    ai_btn = st.button('Check AI Similarity')

if plagiarism_btn:
    st.write("""
    ### Checking for plagiarism or finding similarities...
    """)
    if not text:
        st.write("""
        ### No text found for plagiarism check or finding similarities.
        """)
        st.stop()
    
    if option == 'Find similarities between files':
        similarities = get_similarity_list(texts, filenames)
        df = pd.DataFrame(similarities, columns=['File 1', 'File 2', 'Similarity'])
        df = df.sort_values(by=['Similarity'], ascending=False)
        # Plotting interactive graphs
        plot_scatter(df)
        plot_line(df)
        plot_bar(df)
        plot_pie(df)
        plot_box(df)
        plot_histogram(df)
        plot_3d_scatter(df)
        plot_violin(df)
    else:
        sentences = get_sentences(text)
        urls = []
        progress = st.progress(0)
        for idx, sentence in enumerate(sentences, start=1):
            urls.append(get_url(sentence))
            progress.progress(int(idx / len(sentences) * 100))

        if None in urls:
            st.write("### No plagiarism detected!")
            st.stop()

        similarity_list = get_similarity_list2(text, urls)
        df = pd.DataFrame({'Sentence': sentences, 'URL': urls, 'Similarity': similarity_list})
        df = df.sort_values(by=['Similarity'], ascending=True)

    df = df.reset_index(drop=True)
    
    # Make URLs clickable in the DataFrame
    if 'URL' in df.columns:
        df['URL'] = df['URL'].apply(lambda x: '<a href="{}">{}</a>'.format(x, x) if x else '')
    
    # Center align URL column header
    df_html = df.to_html(escape=False)
    if 'URL' in df.columns:
        df_html = df_html.replace('<th>URL</th>', '<th style="text-align: center;">URL</th>')
    # Show plagiarism results
    st.write(df_html, unsafe_allow_html=True)


    # ---------------- AI SIMILARITY CHECK ----------------
    if ai_btn:
        if not text:
            st.warning("Please enter or upload some text first.")
            st.stop()

        st.write("### Checking AI-generated text probability...")

        result = detect_ai_similarity(text)

        st.metric("AI Similarity Confidence", f"{result['confidence']}%")

        if "fake" in result['label'].lower() or "generated" in result['label'].lower():
            st.warning(f"Likely AI-generated text ({result['confidence']}%)")
        else:
            st.success(f"Likely human-written text ({result['confidence']}%)")