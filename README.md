# Lectora — AI Study Assistant

> An AI-powered web application that turns lecture slides into an interactive study resource.
> Built for DSAI 4201: Selected Topics — March 2026

**Diala Hussein (60104647) · Shaimah Mohammed (60104699)**

---

## What It Does

Upload any lecture file (PDF or PPTX) and Lectora gives you three tools to study from it:

- **Question Answering** — Ask anything about the lecture and get a direct, grounded answer
- **Lecture Summary** — Generate a clean numbered summary of the key points
- **MCQ Quiz** — Auto-generated multiple-choice questions with live feedback and scoring

Everything runs in a browser. No setup beyond installing the requirements.

---

## How It Works

Lectora uses a **Retrieval-Augmented Generation (RAG)** architecture:

1. The lecture file is uploaded and text is extracted and cleaned locally
2. When a query is made, keyword scoring identifies the most relevant sections — no API call yet
3. Only those sections are sent to the **Groq API (LLaMA 3.1 8B)** along with the query
4. The model returns a grounded, accurate response

This keeps API usage minimal and responses fast.

---

## Requirements

- Python 3.10 or higher
- A free Groq API key (see below)

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/Diala-Hussein/Lectora-AI-Study-Assistant.git
cd Lectora-AI-Study-Assistant
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Set up your API key**

Copy the example env file:

```bash
cp .env.example .env
```

Then open `.env` and replace the placeholder with your real Groq key:

```
GROQ_API_KEY=your-actual-key-here
```

---

## Getting a Free Groq API Key

1. Go to **console.groq.com**
2. Sign up with Google or email — no credit card required
3. Click **API Keys → Create API Key**
4. Copy the key (it starts with `gsk_`) — it is only shown once
5. Paste it into your `.env` file

**Free tier:** 14,400 requests/day — more than enough for any study session.

---

## Running the App

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

## Project Structure

```
Lectora-AI-Study-Assistant/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .env.example              # API key template
├── core/
│   ├── qa_engine.py          # Question answering with local retrieval
│   ├── summarizer.py         # Lecture summarisation
│   ├── mcq_generator.py      # MCQ quiz generation
│   └── text_processor.py     # Text cleaning and chunking pipeline
├── extractors/
│   ├── pdf_extractor.py      # PDF text extraction
│   └── pptx_extractor.py     # PowerPoint text extraction
├── utils/
│   ├── openai_client.py      # Groq API client
│   └── similarity.py         # Keyword scoring utilities
├── sample_docs/              # Sample lecture files for testing
├── screenshots/              # App screenshots
└── report/                   # Project report
```

---

## Sample Documents

The `sample_docs/` folder contains lecture files you can use to test the app immediately after setup.

---

## Screenshots

See the `screenshots/` folder for images of the app in action.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Web Interface | Streamlit |
| Language Model | Groq API — LLaMA 3.1 8B Instant |
| PDF Extraction | pdfplumber / PyPDF2 |
| PPTX Extraction | python-pptx |
| API Key Management | python-dotenv |

---

## Course

DSAI 4201: Selected Topics — Spring 2026
