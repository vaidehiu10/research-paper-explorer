import re
from collections import Counter
from io import BytesIO

import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Research Paper Explorer", page_icon="📚", layout="wide")

st.markdown("""
<style>
    .main {background: #f7f8fc;}
    .hero {padding: 1.4rem 1.6rem; border-radius: 18px; background: linear-gradient(120deg,#14213d,#315c8c); color: white;}
    .hero h1 {color: white; margin-bottom: .25rem;}
    .hero p {color: #e7eef8; margin-bottom: 0;}
    .metric-card {padding: 1rem; border: 1px solid #e5e9f2; border-radius: 14px; background: white;}
    .small-muted {color: #65738b; font-size: .9rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>📚 Research Paper Explorer</h1>
<p>Turn lengthy research PDFs into searchable, structured, easier-to-review knowledge.</p>
</div>
""", unsafe_allow_html=True)

st.write("")
with st.sidebar:
    st.header("About this project")
    st.write("A lightweight NLP tool for exploring research papers. Upload a PDF to extract text, search for relevant passages, and create an extractive summary.")
    st.caption("Privacy note: this demo processes the uploaded PDF during your session. Avoid uploading confidential or sensitive documents.")
    st.divider()
    st.write("**Built with**")
    st.write("Python · Streamlit · PyPDF · TF-IDF · scikit-learn")

uploaded = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])

def extract_pdf(file_bytes):
    reader = PdfReader(BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if txt.strip():
            pages.append((i + 1, txt))
    return pages

def clean_text(text):
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_sentences(text):
    chunks = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text)
    return [s.strip() for s in chunks if len(s.strip()) > 45]

def summarize(text, sentence_count=6):
    sentences = split_sentences(text)
    if not sentences:
        return []
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    stop = set("""the and for that with this from are was were have has had into using use used based these those their there which while where when what how why also can may might our we they them you your""".split())
    freq = Counter(w for w in words if w not in stop)
    if not freq:
        return sentences[:sentence_count]
    max_freq = max(freq.values())
    scores = []
    for idx, sentence in enumerate(sentences):
        tokens = [w for w in re.findall(r"\b[a-zA-Z]{3,}\b", sentence.lower()) if w not in stop]
        score = sum(freq[w] / max_freq for w in tokens) / max(len(tokens), 1)
        # Slight preference for sentences near the beginning, common in abstracts/intros.
        score *= 1.0 / (1.0 + idx * 0.002)
        scores.append((score, idx, sentence))
    chosen = sorted(scores, reverse=True)[:sentence_count]
    return [s for _, _, s in sorted(chosen, key=lambda x: x[1])]

if uploaded:
    try:
        pages = extract_pdf(uploaded.getvalue())
        if not pages:
            st.error("No selectable text was found. This may be a scanned PDF; please try a text-based PDF.")
            st.stop()
        page_texts = [(p, clean_text(t)) for p, t in pages]
        full_text = "\n\n".join(t for _, t in page_texts)
        word_count = len(re.findall(r"\b\w+\b", full_text))
        col1, col2, col3 = st.columns(3)
        col1.metric("Pages with text", len(page_texts))
        col2.metric("Words extracted", f"{word_count:,}")
        col3.metric("Document size", f"{len(uploaded.getvalue()) / 1024:.1f} KB")

        st.success(f"Successfully extracted text from **{uploaded.name}**.")
        tab1, tab2, tab3, tab4 = st.tabs(["✨ Summary", "🔎 Search paper", "🧭 Explore sections", "📄 Full text"])

        with tab1:
            st.subheader("Key-point summary")
            count = st.slider("Number of sentences", 3, 12, 6)
            summary_points = summarize(full_text, count)
            if summary_points:
                for i, sentence in enumerate(summary_points, 1):
                    st.markdown(f"**{i}.** {sentence}")
                st.download_button("Download summary", "\n\n".join(f"{i}. {s}" for i, s in enumerate(summary_points, 1)),
                                   file_name="research_paper_summary.txt", mime="text/plain")
            else:
                st.info("Not enough readable text to create a summary.")

        with tab2:
            st.subheader("Find relevant passages")
            query = st.text_input("What do you want to find?", placeholder="e.g. methodology, dataset, limitations, accuracy")
            if query.strip():
                corpus = [t for _, t in page_texts]
                vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
                try:
                    matrix = vectorizer.fit_transform(corpus + [query])
                    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
                    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
                    shown = 0
                    for idx, score in ranked:
                        if score <= 0 or shown >= 5:
                            continue
                        st.markdown(f"**Page {page_texts[idx][0]} · Relevance {score:.2f}**")
                        snippet = page_texts[idx][1]
                        match = re.search(re.escape(query.split()[0]), snippet, re.IGNORECASE)
                        if match:
                            start = max(0, match.start() - 240)
                            end = min(len(snippet), match.end() + 600)
                            snippet = ("…" if start else "") + snippet[start:end] + ("…" if end < len(page_texts[idx][1]) else "")
                        else:
                            snippet = snippet[:850] + ("…" if len(snippet) > 850 else "")
                        st.write(snippet)
                        st.divider()
                        shown += 1
                    if shown == 0:
                        st.info("No close matches found. Try broader keywords.")
                except ValueError:
                    st.warning("The document has too little text to search.")

        with tab3:
            st.subheader("Quick section finder")
            labels = {
                "Abstract": r"\babstract\b",
                "Introduction": r"\bintroduction\b",
                "Related work / Literature review": r"\b(related work|literature review)\b",
                "Methodology": r"\b(methodology|methods|materials and methods)\b",
                "Results": r"\b(results|findings)\b",
                "Discussion": r"\bdiscussion\b",
                "Conclusion": r"\b(conclusion|concluding remarks)\b",
                "Limitations": r"\b(limitations|future work)\b",
            }
            found = []
            for label, pattern in labels.items():
                for page_no, text in page_texts:
                    m = re.search(pattern, text, re.IGNORECASE)
                    if m:
                        found.append((label, page_no, text[m.start():m.start()+700]))
                        break
            if found:
                for label, page_no, snippet in found:
                    with st.expander(f"{label} · around page {page_no}"):
                        st.write(snippet + ("…" if len(snippet) >= 700 else ""))
            else:
                st.info("No standard section headings were detected. Try the search tab.")

        with tab4:
            st.subheader("Extracted document text")
            st.text_area("Text", full_text, height=420)
            st.download_button("Download extracted text", full_text, file_name="extracted_research_paper.txt", mime="text/plain")
    except Exception as exc:
        st.error(f"Could not read this PDF: {exc}")
else:
    st.info("👆 Upload a PDF to get started.")
    st.subheader("What you can do")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### ✨ Summarize")
        st.write("Create a concise extractive summary from the paper.")
    with c2:
        st.markdown("### 🔎 Search")
        st.write("Find pages and passages relevant to a question or keyword.")
    with c3:
        st.markdown("### 🧭 Navigate")
        st.write("Locate common research sections such as methods, results, and limitations.")

st.divider()
st.caption("Research Paper Explorer · Academic productivity demo · Summaries are automated and should be checked against the original paper.")
