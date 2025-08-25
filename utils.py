from urllib.parse import quote_plus
from dotenv import load_dotenv
import requests
import os
from fastapi import FastAPI, HTTPException
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq  # ✅ Using Groq everywhere
from langchain_core.messages import SystemMessage, HumanMessage
from datetime import datetime
from elevenlabs import ElevenLabs
import certifi
import ollama
from pathlib import Path
from gtts import gTTS

# Ensure SSL uses certifi bundle
os.environ["SSL_CERT_FILE"] = certifi.where()

load_dotenv()


class MCPOverloadedError(Exception):
    """Custom exception for MCP service overloads"""
    pass


# ----------------------------
# Utility functions
# ----------------------------
def generate_valid_news_url(keyword: str) -> str:
    q = quote_plus(keyword)
    return f"https://news.google.com/search?q={q}&tbs=sbd:1"


def generate_news_urls_to_scrape(list_of_keywords):
    return {kw: generate_valid_news_url(kw) for kw in list_of_keywords}


def scrape_with_brightdata(url: str) -> str:
    headers = {
        "Authorization": f"Bearer {os.getenv('BRIGHTDATA_API_TOKEN')}",
        "Content-Type": "application/json"
    }
    payload = {
        "zone": os.getenv('WEB_UNLOCKER_ZONE'),
        "url": url,
        "format": "raw"
    }
    try:
        response = requests.post("https://api.brightdata.com/request", json=payload, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"BrightData error: {str(e)}")


def clean_html_to_text(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator="\n").strip()


def extract_headlines(cleaned_text: str) -> str:
    headlines = []
    current_block = []
    lines = [line.strip() for line in cleaned_text.split("\n") if line.strip()]

    for line in lines:
        if line == "More":
            if current_block:
                headlines.append(current_block[0])
                current_block = []
        else:
            current_block.append(line)

    if current_block:
        headlines.append(current_block[0])

    return "\n".join(headlines)


def summarize_with_ollama(headlines) -> str:
    prompt = f"""You are my personal news editor. Summarize these headlines into a TV news script for me.
Focus on important headlines. The output will be read by a podcaster/newscaster:
{headlines}
News Script:"""

    try:
        client = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        response = client.generate(
            model="llama3.2",
            prompt=prompt,
            options={"temperature": 0.4, "max_tokens": 800},
            stream=False
        )
        return response["response"]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ollama error: {str(e)}")


# ----------------------------
# Groq-powered summarizers
# ----------------------------
def generate_broadcast_news(api_key, news_data, reddit_data, topics):
    system_prompt = """
    You are broadcast_news_writer, a professional virtual news reporter. Generate natural, TTS-ready news reports.

    Rules:
    - News first ("According to official reports...")
    - Then Reddit ("Meanwhile, online discussions...")
    - Skip missing sources
    - Keep each topic 60-120 seconds
    - Use transitions
    - Neutral tone, but highlight sentiment
    - End with "To wrap up this segment..."
    """

    try:
        topic_blocks = []
        for topic in topics:
            news_content = news_data.get("news_analysis", {}).get(topic, "")
            reddit_content = reddit_data.get("reddit_analysis", {}).get(topic, "")
            context = []
            if news_content:
                context.append(f"OFFICIAL NEWS CONTENT:\n{news_content}")
            if reddit_content:
                context.append(f"REDDIT DISCUSSION CONTENT:\n{reddit_content}")
            if context:
                topic_blocks.append(f"TOPIC: {topic}\n\n" + "\n\n".join(context))

        user_prompt = (
            "Create broadcast segments for these topics:\n\n" +
            "\n\n--- NEW TOPIC ---\n\n".join(topic_blocks)
        )

        llm = ChatGroq(
            model="llama-3.1-8b-instant",  # ✅ good for longer content
            api_key=api_key,
            temperature=0.3,
            max_tokens=4000,
        )

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

        return response.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Groq error: {str(e)}")


def summarize_with_news_script(api_key: str, headlines: str) -> str:
    system_prompt = """
You are my personal news editor and scriptwriter for a news podcast.
Turn raw headlines into a clean, professional, TTS-friendly news script.
- No special symbols, emojis, or markdown
- No preamble like "Here’s your summary"
- Write clear spoken-language paragraphs
- Keep formal broadcast-style tone
- Focus on important headlines
- Output only the clean script
"""
    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",  # ✅ faster than mixtral
            api_key=api_key,
            temperature=0.4,
            max_tokens=1000
        )

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=headlines)
        ])

        return response.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq error: {str(e)}")


# ----------------------------
# Text-to-Speech
# ----------------------------
def text_to_audio_elevenlabs_sdk(
    text: str,
    voice_id: str = "JBFqnCBsd6RMkjVDRZzb",
    model_id: str = "eleven_multilingual_v2",
    output_format: str = "mp3_44100_128",
    output_dir: str = "audio",
    api_key: str = None
) -> str:
    try:
        api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ElevenLabs API key is required.")

        client = ElevenLabs(api_key=api_key)
        audio_stream = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format
        )

        os.makedirs(output_dir, exist_ok=True)
        filename = f"tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "wb") as f:
            for chunk in audio_stream:
                f.write(chunk)

        return filepath
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ElevenLabs error: {str(e)}")


AUDIO_DIR = Path("audio")
AUDIO_DIR.mkdir(exist_ok=True)

def tts_to_audio(text: str, language: str = "en") -> str:
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = AUDIO_DIR / f"tts_{timestamp}.mp3"
        tts = gTTS(text=text, lang=language, slow=False)
        tts.save(str(filename))
        return str(filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"gTTS error: {str(e)}")
