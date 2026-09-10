# Academic Document Intelligence

Academic Document Intelligence is a Streamlit app for reviewing academic documents with explainable AI-writing signals and document-similarity evidence. Upload a PDF, DOCX, or TXT file to inspect paragraph-level risk, compare it with a local reference corpus, and export a PDF report.

> This project is a writing-support and research demonstration tool. Its AI-writing and similarity scores are estimates, not proof of authorship or plagiarism.

## Repository details

- **Suggested repository name:** `academic-document-intelligence`
- **Suggested description:** `Streamlit app for explainable AI-writing analysis, academic document similarity, paragraph-level evidence, and PDF reports.`
- **Suggested topics:** `streamlit`, `python`, `nlp`, `academic-writing`, `similarity-detection`, `ai-detection`, `document-analysis`

## What it does

- Extracts and cleans text from PDF, DOCX, and TXT files.
- Shows document word count, paragraph count, and a paragraph-level risk map.
- Estimates AI-writing probability with `openai-community/roberta-base-openai-detector`.
- Compares documents using TF-IDF cosine similarity, MinHash Jaccard similarity, and Sentence-BERT semantic similarity.
- Searches a local `corpus/` directory for the strongest reference matches.
- Provides a lightweight humanization fallback and an optional Hugging Face paraphrase model.
- Generates downloadable PDF report summaries.

## How it works

The primary entry point is `app.py`. On first launch, Hugging Face model files are downloaded and cached locally:

- `sentence-transformers/all-MiniLM-L6-v2` for semantic similarity
- `openai-community/roberta-base-openai-detector` for AI-writing signals

The older `app2.py` is retained as an experimental prototype and is not the recommended entry point.

## Requirements

- Python 3.10 or newer
- Internet access on first launch so the Hugging Face models can be downloaded
- Sufficient disk space for the Python packages and cached models

## Installation

### Windows PowerShell

```powershell
git clone https://github.com/AvinashCreates/academic-document-intelligence.git
cd academic-document-intelligence
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run the app with the virtual-environment executable directly:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

### macOS or Linux

```bash
git clone https://github.com/AvinashCreates/academic-document-intelligence.git
cd academic-document-intelligence
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the application

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

To use another port:

```bash
streamlit run app.py --server.port 8502
```

## Use a local comparison corpus

Create a `corpus/` directory in the repository root and add reference PDF, DOCX, or TXT files. The Similarity Evidence tab compares the uploaded document with every supported file in that directory.

```powershell
mkdir corpus
```

Do not commit private papers, student submissions, copyrighted datasets, downloaded model files, or other sensitive material. The included `.gitignore` excludes the local corpus directory by default.

## Typical workflow

1. Start the app with `streamlit run app.py`.
2. Upload a PDF, DOCX, or TXT document.
3. Review the document summary and paragraph risk map.
4. Inspect AI-writing indicators and suggested revisions.
5. Open Similarity Evidence to review local-corpus matches.
6. Use Humanize Document for an optional rewrite draft.
7. Generate and download a PDF report when needed.

## Limitations and responsible use

- AI-writing detectors can produce false positives and false negatives, especially for edited, translated, technical, or non-native writing.
- Similarity is calculated only against the documents available in the local corpus; it is not a web-wide plagiarism search.
- PDF extraction quality depends on whether the PDF contains selectable text. Scanned PDFs may require OCR before analysis.
- Humanization output must be reviewed by the author for meaning, citations, and academic integrity.
- Do not use the scores as the sole basis for grading, disciplinary action, or authorship decisions.

## Troubleshooting

### The app fails during startup

Confirm that the virtual environment is active and reinstall the pinned dependency:

```bash
python -m pip install -r requirements.txt
python -m pip install "starlette==0.46.2"
```

### Models fail to download

Check the network connection, confirm that the Hugging Face model pages are accessible, and run the application again. Model downloads are cached after a successful launch.

### A port is already in use

Start Streamlit on another port, for example `streamlit run app.py --server.port 8502`.

## License

This project is available under the MIT License. See [LICENSE](LICENSE).

## Contributing

Issues and pull requests are welcome. Before opening a pull request, describe the user-facing behavior changed and include reproducible steps for testing it locally.
