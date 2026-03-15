import streamlit as st
import os
from core.text_processor import LectureTextProcessor
from core.qa_engine import QAEngine
from core.summarizer import TextRankSummarizer
from core.mcq_generator import MCQGenerator
from extractors.pdf_extractor import PDFExtractor
from extractors.pptx_extractor import PPTXExtractor
from utils.openai_client import set_api_key, validate_api_key

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Lectora — AI Study Assistant",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

:root {
    --ink:        #0f0e0c;
    --paper:      #f5f0e8;
    --paper-mid:  #ede7d9;
    --amber:      #c8853a;
    --amber-soft: #e8a85a;
    --muted:      #7a7060;
    --rule:       #d4ccc0;
    --white:      #ffffff;
    --radius:     6px;
    --shadow:     0 2px 16px rgba(15,14,12,0.10);
    --font-serif: 'DM Serif Display', Georgia, serif;
    --font-sans:  'DM Sans', system-ui, sans-serif;
}
html, body, [class*="css"] { font-family: var(--font-sans) !important; background-color: var(--paper) !important; color: var(--ink) !important; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.block-container { padding: 2rem 3rem 4rem !important; max-width: 1200px !important; }

/* MASTHEAD */
.masthead { display: flex; align-items: baseline; gap: 1.2rem; padding: 2.4rem 0 1rem; border-bottom: 2px solid var(--ink); margin-bottom: 2.4rem; }
.masthead-title { font-family: var(--font-serif) !important; font-size: 2.8rem !important; font-weight: 400 !important; letter-spacing: -0.02em; color: var(--ink) !important; margin: 0 !important; line-height: 1 !important; }
.masthead-sub { font-size: 0.82rem; letter-spacing: 0.18em; text-transform: uppercase; color: var(--muted); padding-bottom: 2px; }
.masthead-rule { flex: 1; height: 1px; background: var(--ink); opacity: 0.15; }

/* TAGLINE */
.tagline-strip { display: flex; gap: 2.4rem; margin-bottom: 2.8rem; padding: 1rem 1.4rem; background: var(--paper-mid); border-radius: var(--radius); border-left: 3px solid var(--amber); }
.tagline-item { font-size: 0.82rem; letter-spacing: 0.06em; color: var(--muted); }
.tagline-item strong { color: var(--ink); font-weight: 500; }

/* SIDEBAR */
[data-testid="stSidebar"] { background: var(--ink) !important; }
[data-testid="stSidebar"] * { color: var(--paper) !important; }
[data-testid="stSidebar"] .sidebar-logo { font-family: var(--font-serif); font-size: 1.5rem; color: var(--amber-soft) !important; padding: 1.6rem 1rem 0.4rem; display: block; }
[data-testid="stSidebar"] hr { border-color: rgba(245,240,232,0.15) !important; margin: 0.8rem 0 !important; }
[data-testid="stSidebar"] .sidebar-section { font-size: 0.7rem; letter-spacing: 0.18em; text-transform: uppercase; color: rgba(245,240,232,0.45) !important; padding: 1rem 1rem 0.3rem; }
[data-testid="stSidebar"] .sidebar-stat { font-size: 0.9rem; padding: 0.2rem 1rem; }
[data-testid="stSidebar"] .sidebar-stat span { color: var(--amber-soft) !important; font-weight: 500; }

/* KEY INPUT in sidebar */
[data-testid="stSidebar"] input[type="password"],
[data-testid="stSidebar"] input[type="text"] {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    color: var(--paper) !important;
    border-radius: var(--radius) !important;
    font-size: 0.82rem !important;
}
[data-testid="stSidebar"] .stTextInput label { color: rgba(245,240,232,0.6) !important; font-size: 0.75rem !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; }

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.08) !important;
    border: 2px dashed rgba(232,168,90,0.6) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--amber-soft) !important;
    background: rgba(255,255,255,0.12) !important;
}
[data-testid="stFileUploader"] * { color: rgba(245,240,232,0.85) !important; }
[data-testid="stFileUploader"] p { color: rgba(245,240,232,0.7) !important; font-size: 0.85rem !important; }
[data-testid="stFileUploader"] button {
    background: var(--amber) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius) !important;
    padding: 0.4rem 1.1rem !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    cursor: pointer !important;
}
[data-testid="stFileUploader"] button:hover {
    background: #b5732e !important;
}

/* BUTTONS */
.stButton > button { font-family: var(--font-sans) !important; font-size: 0.82rem !important; font-weight: 500 !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; background: var(--ink) !important; color: var(--paper) !important; border: none !important; border-radius: var(--radius) !important; padding: 0.55rem 1.4rem !important; transition: background 0.2s, transform 0.1s !important; }
.stButton > button:hover { background: var(--amber) !important; transform: translateY(-1px) !important; }
.stButton > button[kind="primary"] { background: var(--amber) !important; color: var(--white) !important; }
.stButton > button[kind="primary"]:hover { background: #b5732e !important; }

/* TABS */
.stTabs [data-baseweb="tab-list"] { gap: 0 !important; border-bottom: 2px solid var(--rule) !important; background: transparent !important; }
.stTabs [data-baseweb="tab"] { font-family: var(--font-sans) !important; font-size: 0.8rem !important; font-weight: 500 !important; letter-spacing: 0.12em !important; text-transform: uppercase !important; color: var(--muted) !important; padding: 0.7rem 1.6rem !important; border: none !important; border-bottom: 2px solid transparent !important; background: transparent !important; margin-bottom: -2px !important; }
.stTabs [aria-selected="true"] { color: var(--ink) !important; border-bottom-color: var(--amber) !important; background: transparent !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.8rem !important; }

/* EXPANDERS */
[data-testid="stExpander"] { background: var(--white) !important; border: 1px solid var(--rule) !important; border-radius: var(--radius) !important; margin-bottom: 0.7rem !important; box-shadow: var(--shadow) !important; }
[data-testid="stExpander"] summary { font-size: 0.88rem !important; font-weight: 500 !important; color: var(--ink) !important; padding: 0.85rem 1.1rem !important; }

/* TEXT INPUT */
.stTextInput input { font-family: var(--font-sans) !important; background: var(--white) !important; border: 1.5px solid var(--rule) !important; border-radius: var(--radius) !important; color: var(--ink) !important; padding: 0.6rem 0.9rem !important; }
.stTextInput input:focus { border-color: var(--amber) !important; box-shadow: 0 0 0 3px rgba(200,133,58,0.12) !important; }

/* CARDS */
.answer-card { background: var(--white); border: 1px solid var(--rule); border-radius: var(--radius); padding: 1.4rem 1.6rem; margin-bottom: 0.8rem; box-shadow: var(--shadow); position: relative; }
.answer-card::before { content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: var(--amber); border-radius: var(--radius) 0 0 var(--radius); }
.answer-text { font-size: 0.97rem; line-height: 1.75; color: var(--ink); }

/* SUMMARY */
.summary-item { display: flex; gap: 1rem; padding: 0.9rem 0; border-bottom: 1px solid var(--rule); align-items: flex-start; }
.summary-item:last-child { border-bottom: none; }
.summary-num { font-family: var(--font-serif); font-size: 1.3rem; color: var(--amber); line-height: 1; min-width: 1.4rem; padding-top: 0.05rem; }
.summary-text { font-size: 0.94rem; line-height: 1.65; color: var(--ink); }

/* MCQ */
.mcq-card { background: var(--white); border: 1px solid var(--rule); border-radius: var(--radius); padding: 1.3rem 1.5rem 1rem; margin-bottom: 1rem; box-shadow: var(--shadow); }
.mcq-label { font-size: 0.7rem; letter-spacing: 0.16em; text-transform: uppercase; color: var(--amber); margin-bottom: 0.4rem; font-weight: 500; }
.mcq-question { font-size: 0.97rem; font-weight: 500; line-height: 1.55; color: var(--ink); margin-bottom: 1rem; }

/* SECTION HEADERS */
.section-header { font-family: var(--font-serif); font-size: 1.6rem; color: var(--ink); margin-bottom: 0.3rem; font-weight: 400; }
.section-meta { font-size: 0.8rem; letter-spacing: 0.06em; color: var(--muted); margin-bottom: 1.6rem; }

/* STATUS */
.status-ready { display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; color: #4a7a4d; background: #edf6ee; border: 1px solid #b5d6b8; padding: 0.28rem 0.7rem; border-radius: 99px; }
.status-dot { width: 6px; height: 6px; border-radius: 50%; background: #4a7a4d; display: inline-block; }
.status-key { display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; color: #9a6820; background: #fef6ec; border: 1px solid #f0d0a0; padding: 0.28rem 0.7rem; border-radius: 99px; }

/* INFO TILES */
.info-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.8rem; margin-bottom: 1.6rem; }
.info-tile { background: var(--white); border: 1px solid var(--rule); border-radius: var(--radius); padding: 0.9rem 1rem; text-align: center; }
.info-tile-val { font-family: var(--font-serif); font-size: 1.6rem; color: var(--amber); line-height: 1; }
.info-tile-lbl { font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); margin-top: 0.25rem; }

/* EMPTY STATE */
.empty-state { text-align: center; padding: 3.5rem 2rem; color: var(--muted); }
.empty-icon { font-size: 2.4rem; margin-bottom: 0.8rem; opacity: 0.6; }
.empty-text { font-size: 0.9rem; line-height: 1.6; }

/* API warning box */
.api-warning { background: #fef6ec; border: 1px solid #f0d0a0; border-radius: var(--radius); padding: 1rem 1.2rem; font-size: 0.88rem; color: #7a5010; margin-bottom: 1.4rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# STUDY ASSISTANT
# ─────────────────────────────────────────────
class StudyAssistant:
    def __init__(self):
        self.processor      = LectureTextProcessor()
        self.qa_engine      = QAEngine()
        self.summarizer     = TextRankSummarizer()
        self.mcq_gen        = MCQGenerator()
        self.sentences      = []
        self.processed_text = ""
        self.file_name      = ""
        self.char_count     = 0
        self.sentence_count = 0

    def process_file(self, uploaded_file) -> bool:
        try:
            uploaded_file.seek(0)
            ext = os.path.splitext(uploaded_file.name)[1].lower()
            if ext == ".pdf":
                extractor = PDFExtractor()
            elif ext == ".pptx":
                extractor = PPTXExtractor()
            else:
                st.error("Only PDF and PPTX files are supported.")
                return False

            raw_text = extractor.extract(uploaded_file)
            if not raw_text or len(raw_text.strip()) < 50:
                st.error("The file appears empty or contains no readable text.")
                return False

            cleaned_text   = self.processor.clean_slide_text(raw_text)
            self.sentences = self.processor.preprocess_sentences(cleaned_text)
            sent_texts     = [s for s, _ in self.sentences]

            if not sent_texts:
                st.error("No usable sentences found in the document.")
                return False

            self.qa_engine.fit(sent_texts)
            self.processed_text = cleaned_text
            self.file_name      = uploaded_file.name
            self.char_count     = len(cleaned_text)
            self.sentence_count = len(sent_texts)
            return True

        except Exception as e:
            st.error(f"Processing error: {e}")
            return False


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "assistant"  not in st.session_state:
    st.session_state.assistant  = StudyAssistant()
if "processed"  not in st.session_state:
    st.session_state.processed  = False
if "api_key_ok" not in st.session_state:
    st.session_state.api_key_ok = bool(os.getenv("GROQ_API_KEY", ""))
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = []
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}

assistant = st.session_state.assistant


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<span class="sidebar-logo">Lectora</span>', unsafe_allow_html=True)
    st.markdown('<hr/>', unsafe_allow_html=True)

    # ── API Key ──────────────────────────────
    st.markdown('<div class="sidebar-section">OpenAI API Key</div>', unsafe_allow_html=True)
    api_key_input = st.text_input(
        "API Key",
        type="password",
        placeholder="gsk_...",
        label_visibility="collapsed",
        help="Get your free key at console.groq.com",
    )
    if api_key_input:
        if validate_api_key(api_key_input):
            set_api_key(api_key_input)
            st.session_state.api_key_ok = True
            st.markdown('<div class="sidebar-stat" style="color:#6db870 !important;">✓ Key accepted</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="sidebar-stat" style="color:#e07070 !important;">✗ Invalid key format</div>', unsafe_allow_html=True)
    elif st.session_state.api_key_ok:
        st.markdown('<div class="sidebar-stat" style="color:#6db870 !important;">✓ Key loaded from environment</div>', unsafe_allow_html=True)

    st.markdown('<hr/>', unsafe_allow_html=True)

    # ── Upload ───────────────────────────────
    st.markdown('<div class="sidebar-section">Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Lecture file",
        type=["pdf", "pptx"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        size_kb = uploaded_file.size / 1024
        st.markdown(f"""
        <div class="sidebar-stat">📄 {uploaded_file.name[:26]}{"…" if len(uploaded_file.name)>26 else ""}</div>
        <div class="sidebar-stat">{size_kb:.1f} KB</div>
        """, unsafe_allow_html=True)

        btn_disabled = not st.session_state.api_key_ok
        if st.button("Analyse Lecture", type="primary", use_container_width=True, disabled=btn_disabled):
            with st.spinner("Processing…"):
                ok = assistant.process_file(uploaded_file)
            if ok:
                st.session_state.processed = True
                st.rerun()

        if btn_disabled:
            st.markdown('<div class="sidebar-stat" style="color:#e0a070 !important;font-size:0.78rem !important;">↑ Enter your API key first</div>', unsafe_allow_html=True)

    st.markdown('<hr/>', unsafe_allow_html=True)

    if st.session_state.processed:
        st.markdown('<div class="sidebar-section">Document Stats</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sidebar-stat">Characters <span>{assistant.char_count:,}</span></div>
        <div class="sidebar-stat">Sentences <span>{assistant.sentence_count}</span></div>
        <div class="sidebar-stat">Type <span>{os.path.splitext(assistant.file_name)[1].upper()}</span></div>
        """, unsafe_allow_html=True)

    st.markdown('<hr/>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-section">Cost Guide</div>
    <div class="sidebar-stat" style="line-height:1.7;padding-top:0.2rem;font-size:0.8rem !important;">
      Q&amp;A answer &nbsp;&nbsp;<span>Free</span><br>
      Summary &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span>Free</span><br>
      10 MCQs &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span>Free</span><br>
      Daily limit &nbsp;&nbsp;&nbsp;&nbsp;<span>14,400 req</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MASTHEAD
# ─────────────────────────────────────────────
st.markdown("""
<div class="masthead">
  <h1 class="masthead-title">Lectora</h1>
  <span class="masthead-sub">AI Study Assistant</span>
  <div class="masthead-rule"></div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
if not st.session_state.api_key_ok:
    st.markdown("""
    <div class="api-warning">
      <strong>API key required.</strong>
      Paste your Groq key in the sidebar to get started.
      Get one free at <strong>console.groq.com → Get API Key</strong>.
      Completely free — 14,400 requests/day.
    </div>
    """, unsafe_allow_html=True)

if not st.session_state.processed:
    st.markdown("""
    <div class="tagline-strip">
      <div class="tagline-item"><strong>Groq Q&amp;A</strong> — Real answers from your slides</div>
      <div class="tagline-item"><strong>Abstractive Summaries</strong> — Not just copied sentences</div>
      <div class="tagline-item"><strong>Smart MCQ Quizzes</strong> — Plausible distractors</div>
      <div class="tagline-item"><strong>100% Free</strong> — Groq free tier, no cost at all</div>
    </div>
    <div class="empty-state">
      <div class="empty-icon">📖</div>
      <div class="empty-text">
        Enter your API key, then upload a PDF or PPTX lecture file<br>
        using the sidebar to begin your study session.
      </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Status + stats ──
    st.markdown(f"""
    <div style="display:flex;gap:0.7rem;align-items:center;margin-bottom:1.8rem;flex-wrap:wrap;">
      <span class="status-ready"><span class="status-dot"></span>Ready — {assistant.file_name}</span>
      <span class="status-key">Llama 3.1 8B</span>
    </div>
    <div class="info-grid">
      <div class="info-tile"><div class="info-tile-val">{assistant.char_count:,}</div><div class="info-tile-lbl">Characters</div></div>
      <div class="info-tile"><div class="info-tile-val">{assistant.sentence_count}</div><div class="info-tile-lbl">Sentences</div></div>
      <div class="info-tile"><div class="info-tile-val">{os.path.splitext(assistant.file_name)[1].upper()}</div><div class="info-tile-lbl">Format</div></div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Ask Questions", "Summary", "Quiz"])

    # ════════════════════════════════════════════
    # Q&A
    # ════════════════════════════════════════════
    with tab1:
        st.markdown('<div class="section-header">Ask a Question</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-meta">Groq reads the most relevant parts of your lecture and gives a direct answer.</div>', unsafe_allow_html=True)

        question = st.text_input(
            "question",
            placeholder="e.g. What is gradient descent? How does backpropagation work?",
            label_visibility="collapsed",
        )

        if st.button("Get Answer", type="primary"):
            if not question.strip():
                st.warning("Please enter a question first.")
            else:
                with st.spinner("Reading lecture and generating answer…"):
                    try:
                        answers = assistant.qa_engine.answer(question, top_k=3)
                        if answers:
                            for answer_text, score in answers:
                                st.markdown(f"""
                                <div class="answer-card">
                                  <div class="answer-text">{answer_text}</div>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No relevant content found. Try rephrasing.")
                    except Exception as e:
                        st.error(f"Error: {e}")

    # ════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════
    with tab2:
        st.markdown('<div class="section-header">Lecture Summary</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-meta">Groq reads the full lecture and writes a clean abstractive summary — not just copied sentences.</div>', unsafe_allow_html=True)

        col_s, col_btn = st.columns([3, 1])
        with col_s:
            num_sent = st.slider("Number of summary points", 3, 12, 6)
        with col_btn:
            st.markdown("<div style='padding-top:1.65rem;'></div>", unsafe_allow_html=True)
            gen_summary = st.button("Generate", type="primary", use_container_width=True)

        if gen_summary:
            with st.spinner("Summarising lecture…"):
                try:
                    sentence_texts = [s for s, _ in assistant.sentences]
                    summary = assistant.summarizer.summarize(sentence_texts, num_sent)

                    if summary:
                        items_html = "".join(
                            f'<div class="summary-item"><div class="summary-num">{i}</div><div class="summary-text">{s}</div></div>'
                            for i, s in enumerate(summary, 1)
                        )
                        st.markdown(f"""
                        <div style="background:var(--white);border:1px solid var(--rule);border-radius:var(--radius);padding:1.2rem 1.4rem;box-shadow:var(--shadow);">
                          {items_html}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("Could not generate a summary from this content.")
                except Exception as e:
                    st.error(f"Error: {e}")

    # ════════════════════════════════════════════
    # QUIZ
    # ════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="section-header">MCQ Quiz</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-meta">Groq generates questions with plausible distractors drawn directly from lecture concepts.</div>', unsafe_allow_html=True)

        col_q, col_btn2 = st.columns([3, 1])
        with col_q:
            num_q = st.slider("Number of questions", 3, 12, 6)
        with col_btn2:
            st.markdown("<div style='padding-top:1.65rem;'></div>", unsafe_allow_html=True)
            gen_quiz = st.button("Generate Quiz", type="primary", use_container_width=True)

        # Generate and store quiz in session state so it survives reruns
        if gen_quiz:
            with st.spinner("Generating questions…"):
                try:
                    mcqs = assistant.mcq_gen.generate_mcqs(assistant.sentences, num_q)
                    if mcqs:
                        st.session_state.quiz_questions = mcqs
                        st.session_state.quiz_answers = {}
                    else:
                        st.warning("No questions could be generated. Try a longer document.")
                except Exception as e:
                    st.error(f"Error: {e}")

        # Render quiz from session state (persists across reruns)
        if st.session_state.quiz_questions:
            mcqs = st.session_state.quiz_questions

            # Score summary at top if any answered
            answered = len(st.session_state.quiz_answers)
            if answered > 0:
                correct = sum(1 for v in st.session_state.quiz_answers.values() if v["correct"])
                st.markdown(f"""
                <div style="background:var(--white);border:1px solid var(--rule);border-radius:var(--radius);
                     padding:0.9rem 1.2rem;margin-bottom:1.4rem;display:flex;gap:2rem;align-items:center;">
                  <div style="font-size:0.8rem;letter-spacing:0.08em;text-transform:uppercase;color:var(--muted);">
                    Progress
                  </div>
                  <div style="font-size:0.95rem;font-weight:500;">
                    {answered} / {len(mcqs)} answered &nbsp;·&nbsp;
                    <span style="color:#4a7a4d;">{correct} correct</span> &nbsp;·&nbsp;
                    <span style="color:#c0392b;">{answered - correct} incorrect</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            for i, mcq in enumerate(mcqs, 1):
                answered_this = i in st.session_state.quiz_answers

                st.markdown(f"""
                <div class="mcq-card">
                  <div class="mcq-label">Question {i} of {len(mcqs)}</div>
                  <div class="mcq-question">{mcq["question"]}</div>
                </div>
                """, unsafe_allow_html=True)

                cols = st.columns(2)
                for j, option in enumerate(mcq["options"]):
                    with cols[j % 2]:
                        # Style answered buttons differently
                        if answered_this:
                            selected_j = st.session_state.quiz_answers[i]["selected"]
                            correct_j  = mcq["correct_index"]
                            if j == correct_j:
                                st.success(f"{chr(65+j)}.  {option}")
                            elif j == selected_j:
                                st.error(f"{chr(65+j)}.  {option}")
                            else:
                                st.button(
                                    f"{chr(65+j)}.  {option}",
                                    key=f"mcq_{i}_{j}_done",
                                    disabled=True,
                                    use_container_width=True,
                                )
                        else:
                            if st.button(
                                f"{chr(65+j)}.  {option}",
                                key=f"mcq_{i}_{j}",
                                use_container_width=True,
                            ):
                                is_correct = (j == mcq["correct_index"])
                                st.session_state.quiz_answers[i] = {
                                    "selected": j,
                                    "correct":  is_correct,
                                }
                                st.rerun()

                # Show explanation after answering
                if answered_this:
                    if st.session_state.quiz_answers[i]["correct"]:
                        st.markdown(
                            "<div style='padding:0.4rem 0.8rem;font-size:0.85rem;color:#4a7a4d;'>✓ Correct!</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"<div style='padding:0.4rem 0.8rem;font-size:0.85rem;color:#c0392b;'>✗ Incorrect — "
                            f"Correct answer: <strong>{mcq['answer']}</strong></div>",
                            unsafe_allow_html=True
                        )

                st.markdown("<div style='margin-bottom:0.8rem;'></div>", unsafe_allow_html=True)

            # Reset button
            if st.button("↺ Reset Quiz", use_container_width=False):
                st.session_state.quiz_questions = []
                st.session_state.quiz_answers = {}
                st.rerun()


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="margin-top:4rem;padding-top:1.2rem;border-top:1px solid var(--rule);display:flex;justify-content:space-between;align-items:center;font-size:0.76rem;color:var(--muted);letter-spacing:0.06em;">
  <span>Lectora — AI Study Assistant</span>
  <span>Powered by Llama 3.1 8B · Free tier</span>
</div>
""", unsafe_allow_html=True)
