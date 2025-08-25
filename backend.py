# backend.py
from __future__ import annotations

import os
import logging
import traceback
import inspect
from pathlib import Path
from typing import Any, Iterable, Literal

import certifi
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Keep TLS verification ON; use certifi bundle
os.environ["SSL_CERT_FILE"] = certifi.where()

# ---- Local modules -----------------------------------------------------------
from models import NewsRequest
from utils import generate_broadcast_news, text_to_audio_elevenlabs_sdk, tts_to_audio
from news_scraper import NewsScraper
from reddit_scraper import scrape_reddit_topics

# ------------------------------------------------------------------------------
# App & logging
# ------------------------------------------------------------------------------
load_dotenv()
app = FastAPI(title="Agentic AI Journalist Backend", version="1.4.0")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "DEBUG"),
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("agentic-ai-journalist")

# Static mount for audio files
os.makedirs("audio", exist_ok=True)
app.mount("/audio", StaticFiles(directory="audio"), name="audio")


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def _require_env(var_names: list[str]) -> None:
    missing = [v for v in var_names if not os.getenv(v)]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Missing required environment variables: {', '.join(missing)}",
        )

# ASCII-only truncation (avoid Unicode ellipsis in headers)
def _truncate(text: str, n: int = 1200) -> str:
    return text if len(text) <= n else text[:n] + " ... <truncated> ..."

# Sanitize any string destined for HTTP headers:
# - remove CR/LF
# - coerce to latin-1 by dropping non-encodable chars
# - strip non-printable control chars
# - limit length
def _header_sanitize(text: str, max_len: int = 900) -> str:
    if not isinstance(text, str):
        text = str(text)
    s = text.replace("\r", " ").replace("\n", " ")
    try:
        s = s.encode("latin-1", "ignore").decode("latin-1")
    except Exception:
        s = s.encode("ascii", "ignore").decode("ascii")
    s = "".join(ch if 32 <= ord(ch) <= 126 else " " for ch in s)
    s = " ".join(s.split())  # collapse excess whitespace
    return s[:max_len]


def _summarise_exception_chain(e: BaseException, depth: int = 0, max_depth: int = 4) -> list[str]:
    msgs: list[str] = []

    subs = getattr(e, "exceptions", None)
    if subs and isinstance(subs, Iterable):
        for child in subs:
            msgs.extend(_summarise_exception_chain(child, depth + 1, max_depth))
        return msgs

    base = f"{type(e).__name__}: {e}"

    try:
        resp = getattr(e, "response", None)
        req = getattr(e, "request", None)
        if resp is not None:
            line = f"{base} | HTTP {getattr(resp, 'status_code', '?')}"
            if req is not None:
                line += f" {getattr(req, 'method', '')} {getattr(req, 'url', '')}"
            try:
                body = resp.text
                if body:
                    line += " | body=" + _truncate(body.replace("\n", " "))
            except Exception:
                pass
            for h in ("x-ratelimit-remaining", "x-ratelimit-reset", "www-authenticate"):
                try:
                    if h in resp.headers:
                        line += f" | {h}={resp.headers[h]}"
                except Exception:
                    pass
            msgs.append(line)
        else:
            msgs.append(base)
    except Exception:
        msgs.append(base)

    tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    tail = "\n".join(tb.strip().splitlines()[-6:])
    msgs.append("TB tail: " + tail.replace("\n", " "))  # keep single-line for header safety

    if depth < max_depth:
        if getattr(e, "__cause__", None):
            msgs.append("Caused by ->")
            msgs.extend(_summarise_exception_chain(e.__cause__, depth + 1, max_depth))
        elif getattr(e, "__context__", None):
            msgs.append("During handling ->")
            msgs.extend(_summarise_exception_chain(e.__context__, depth + 1, max_depth))
    return msgs


def _fallback_summary(news_data: Any, reddit_data: Any, topics: list[str]) -> str:
    def pick_items(blob, max_topics=3, max_per_topic=3):
        items = []
        if isinstance(blob, dict):
            for t_idx, (t, lst) in enumerate(blob.items()):
                if t_idx >= max_topics:
                    break
                if isinstance(lst, list):
                    for x in lst[:max_per_topic]:
                        if isinstance(x, dict):
                            title = x.get("title") or x.get("headline") or ""
                            summ = x.get("summary") or x.get("description") or ""
                            src = x.get("source") or x.get("subreddit") or ""
                            if title or summ:
                                items.append((t, title, summ, src))
        return items[: max_topics * max_per_topic]

    lines = ["Here is a brief round-up from the last two weeks.\n"]

    news_items = pick_items(news_data)
    if news_items:
        lines.append("News highlights:")
        for t, title, summ, src in news_items:
            bit = f"- [{t}] {title}".strip()
            if summ:
                bit += f" — {summ[:160]}".rstrip()
            if src:
                bit += f" ({src})"
            lines.append(bit)

    red_items = pick_items(reddit_data)
    if red_items:
        lines.append("\nReddit discussion snapshots:")
        for t, title, summ, src in red_items:
            bit = f"- [{t}] {title}".strip()
            if summ:
                bit += f" — {summ[:160]}".rstrip()
            if src:
                bit += f" (r/{src})"
            lines.append(bit)

    if not news_items and not red_items:
        lines.append("No structured items were available; providing a topic-based update:")
        for t in topics:
            lines.append(f"- {t}: monitoring recent developments; more details unavailable from sources.")

    lines.append("\nEnd of bulletin.")
    return "\n".join(lines)


# ---------- Flexible TTS fallback adapter ------------------------------------
def _coerce_tts_result_to_path(result: Any, out_dir: str, default_name: str = "news-summary-fallback.mp3") -> str | None:
    os.makedirs(out_dir, exist_ok=True)
    default_path = os.path.join(out_dir, default_name)

    if result is None:
        return None
    if isinstance(result, (str, Path)):
        return str(result)
    if isinstance(result, (bytes, bytearray)):
        with open(default_path, "wb") as f:
            f.write(result)
        return default_path
    if isinstance(result, dict):
        for key in ("path", "file", "filename", "output_path"):
            if key in result and result[key]:
                return str(result[key])
        audio_bytes = result.get("audio")
        if isinstance(audio_bytes, (bytes, bytearray)):
            with open(default_path, "wb") as f:
                f.write(audio_bytes)
            return default_path
    return None


def _try_tts_flexible(text: str, out_dir: str = "audio") -> str | None:
    os.makedirs(out_dir, exist_ok=True)
    try:
        sig = inspect.signature(tts_to_audio)
        param_names = set(sig.parameters.keys())
    except Exception:
        param_names = set()

    target_path = os.path.join(out_dir, "news-summary-fallback.mp3")

    kwargs: dict[str, Any] = {}
    if "text" in param_names:
        kwargs["text"] = text
    if "output_dir" in param_names:
        kwargs["output_dir"] = out_dir
    if "out_dir" in param_names:
        kwargs["out_dir"] = out_dir
    for path_kw in ("output_path", "path", "file_path", "filepath", "filename"):
        if path_kw in param_names:
            kwargs[path_kw] = target_path

    try:
        if "text" in kwargs:
            result = tts_to_audio(**kwargs)
        else:
            result = tts_to_audio(text, **{k: v for k, v in kwargs.items() if k != "text"})
        path = _coerce_tts_result_to_path(result, out_dir)
        if path:
            return path
    except TypeError:
        pass
    except Exception as e:
        logger.debug("Flexible TTS call (kwargs) failed: %r", e)

    try:
        result = tts_to_audio(text)
        path = _coerce_tts_result_to_path(result, out_dir)
        if path:
            return path
    except Exception as e:
        logger.debug("Flexible TTS call (text only) failed: %r", e)

    return None
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Route
# ------------------------------------------------------------------------------
@app.post("/generate-news-audio")
async def generate_news_audio(
    request: NewsRequest,
    return_format: Literal["audio", "json", "both"] = "audio",
):
    """
    Generates a news summary and returns:
      - audio  : MP3 stream (FileResponse)
      - json   : {"summary": "...", "audio_url": "...", "warnings": {...}}
      - both   : same JSON (text + audio_url)
    If Reddit/LLM/TTS fails, we degrade gracefully and expose diagnostics (ASCII-only) in headers or JSON.
    """
    _require_env(["GROQ_API_KEY"])
    if not (os.getenv("ELEVEN_API_KEY") or os.getenv("ELEVENLABS_API_KEY")) and return_format == "audio":
        # Only require ElevenLabs when client demands audio and no fallback produces it
        # We'll still try fallback TTS; this check is a friendly early warning for primary TTS.
        logger.warning("ELEVEN_* key not set; will rely on fallback TTS if needed.")

    warn_headers: dict[str, str] = {}
    results: dict[str, Any] = {}
    source_type = (request.source_type or "both").lower()

    try:
        # ---- News (fail-open)
        if source_type in {"news", "both"}:
            news_scraper = NewsScraper()
            try:
                results["news"] = await news_scraper.scrape_news(request.topics)
            except Exception as e:
                diag = " | ".join(_summarise_exception_chain(e))
                logger.error("News scraping failed: %s", diag)
                warn_headers["X-News-Warning"] = _header_sanitize("news-scrape-failed")
                results["news"] = {}

        # ---- Reddit (fail-open)
        if source_type in {"reddit", "both"}:
            try:
                results["reddit"] = await scrape_reddit_topics(request.topics)
            except Exception as e:
                diag = " | ".join(_summarise_exception_chain(e))
                logger.error("Reddit scraping failed: %s", diag)
                warn_headers["X-Reddit-Warning"] = _header_sanitize("reddit-scrape-failed")
                results["reddit"] = {}

        news_data = results.get("news") or {}
        reddit_data = results.get("reddit") or {}

        # ---- LLM summary (fallback to deterministic text on failure)
        try:
            news_summary = generate_broadcast_news(
                api_key=os.getenv("GROQ_API_KEY"),
                news_data=news_data,
                reddit_data=reddit_data,
                topics=request.topics,
            )
        except Exception as e:
            diag = " | ".join(_summarise_exception_chain(e))
            logger.error("LLM summarisation failed: %s", diag)
            warn_headers["X-LLM-Warning"] = _header_sanitize(_truncate(diag, 400))
            news_summary = _fallback_summary(news_data, reddit_data, request.topics)

        # ---- TTS (only if audio requested)
        audio_path: str | None = None
        if return_format in ("audio", "both"):
            try:
                audio_path = text_to_audio_elevenlabs_sdk(
                    text=news_summary,
                    voice_id="JBFqnCBsd6RMkjVDRZzb",
                    model_id="eleven_multilingual_v2",
                    output_format="mp3_44100_128",
                    output_dir="audio",
                )
            except Exception as e:
                diag = " | ".join(_summarise_exception_chain(e))
                logger.error("ElevenLabs TTS failed: %s", diag)
                warn_headers["X-TTS-Warning"] = _header_sanitize(_truncate(diag, 400))

            # Fallback TTS
            if not audio_path or not Path(audio_path).exists():
                path2 = _try_tts_flexible(text=news_summary, out_dir="audio")
                if path2:
                    audio_path = path2
                    warn_headers["X-TTS-Fallback"] = "1"
                elif return_format == "audio":
                    # Only hard-fail if client demanded audio
                    raise HTTPException(status_code=502, detail="Text-to-Speech failed: no usable audio produced.")

        # ---- Optional audio URL (served via /audio)
        audio_url = None
        if audio_path and Path(audio_path).exists():
            audio_url = f"/audio/{os.path.basename(audio_path)}"

        # ---- Respond
        if return_format == "audio":
            return FileResponse(
                path=audio_path,                 # guaranteed above or raised
                media_type="audio/mpeg",
                filename=os.path.basename(audio_path),
                headers=warn_headers,            # ASCII-safe diagnostics
            )

        # JSON (text + optional audio_url + warnings)
        payload = {
            "summary": news_summary,
            "audio_url": audio_url,             # may be None
            "warnings": warn_headers,           # same diagnostics but in payload
            "topics": request.topics,
            "sources_used": {
                "news": bool(news_data),
                "reddit": bool(reddit_data),
            },
        }
        return JSONResponse(content=payload, status_code=200)

    except HTTPException:
        raise
    except Exception as e:
        diag = " | ".join(_summarise_exception_chain(e))
        logger.error("generate-news-audio failed: %s", diag)
        raise HTTPException(status_code=502, detail=diag)


# ------------------------------------------------------------------------------
# Local dev entry point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend:app",
        host="127.0.0.1",
        port=1234,
        reload=True,
        log_level="debug",
    )
