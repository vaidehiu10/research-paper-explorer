# Research Paper Explorer

A Streamlit + NLP application that helps users review research papers in PDF format.

## Features
- Extracts selectable text from uploaded PDFs
- Generates an extractive summary using sentence scoring
- Searches document pages with TF-IDF similarity
- Detects common academic sections
- Downloads summaries and extracted text

## Run locally
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy free with Streamlit Community Cloud
1. Create a **public GitHub repository** and upload `app.py`, `requirements.txt`, and `README.md`.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Choose **Create app**, select your repository, branch (`main`), and main file path (`app.py`).
4. Click **Deploy**. Streamlit will provide a public `*.streamlit.app` URL.

No API key or secrets are required.

## Limitations
- Works best with PDFs containing selectable text; scanned image-only PDFs need OCR.
- The summary is extractive, not a verified scientific interpretation.
- Search ranks relevant pages; it does not guarantee a complete answer.
- Do not upload confidential documents to a public demo.
