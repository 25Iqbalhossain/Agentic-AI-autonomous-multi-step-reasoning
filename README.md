# 🧠 Agentic AI Autonomous 

An autonomous, multi-step AI pipeline that scrapes live news and Reddit data, cleans and structures HTML content, summarizes it using multiple LLMs via MCP, and transforms it into natural speech using ElevenLabs. Built for clarity, speed, and voice-first delivery.

---

## 🚀 Features

- 🌐 **BrightData Scraping**  
  Scrapes Google News and Reddit topics using BrightData proxies for reliable, real-time access.

- 🧼 **HTML Cleaning with BeautifulSoup**  
  Parses and sanitizes raw HTML content for structured downstream processing.

- 🧠 **Multi-LLM Reasoning via MCP**  
  Uses Groq-hosted LLMs and MCP agents for layered summarization, sentiment extraction, and broadcast-style scripting.

- 🎙️ **Voice Synthesis with ElevenLabs**  
  Converts clean scripts into natural-sounding audio using multilingual ElevenLabs voices.

- ⚙️ **FastAPI Backend**  
  Modular API endpoints for scraping, summarizing, and generating downloadable audio files.

---

## 🧩 Agentic Workflow

1. **Scrape**: BrightData fetches raw HTML from Google News and Reddit.
2. **Clean**: BeautifulSoup extracts readable content.
3. **Summarize**: MCP routes content through multiple LLMs (Groq, Mixtral, LLaMA3).
4. **Speak**: ElevenLabs generates TTS-ready audio segments.
5. **Serve**: FastAPI delivers the final audio via `/generate-news-audio`.

---

## 📦 Tech Stack

- `BrightData` – Proxy-based scraping
- `BeautifulSoup` – HTML parsing
- `LangChain + MCP` – Multi-agent LLM orchestration
- `Groq` – Ultra-fast LLM inference
- `ElevenLabs` – Text-to-speech synthesis
- `FastAPI` – Backend API
- `Python` – Core language

---

## 🔐 Environment Variables

Create a `.env` file with the following keys:

```env
BRIGHTDATA_API_KEY=
GROQ_API_KEY=
ANTHROPIC_API_KEY=
ELEVENLABS_API_KEY=
```
git clone https://github.com/25Iqbalhossain/Agentic-AI-autonomous-multi-step-reasoning.git.
cd Agentic-AI-autonomous-multi-step-reasoning.
python -m venv venv.
source venv/Scripts/activate  # or use . venv/bin/activate on Unix.
pip install -r requirements.txt.

## 📣 Credits
Built by Iqbal Hossain
Inspired by agentic AI workflows and voice-first UX


## 📜 License
MIT License

Let me know if you want to add badges, a demo video link, or GitHub Actions for auto-deployment. This README sets the tone for a serious, modular, and voice-powered AI project.
