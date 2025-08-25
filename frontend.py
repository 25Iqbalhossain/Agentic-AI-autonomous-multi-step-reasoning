# frontend.py
import os
from typing import Literal, Dict, Any, Optional

import streamlit as st
import requests

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
SOURCE_TYPES = Literal["news", "reddit", "both"]
RETURN_FORMAT = Literal["audio", "json", "both"]

# Use env var if you run backend on a different port
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


# ----------------------------------------------------------------------
# Helpers (no diagnostics UI)
# ----------------------------------------------------------------------
def _handle_api_error(response: requests.Response) -> None:
    """Consistent API error handling (no diagnostics displayed)."""
    try:
        detail = response.json().get("detail", "Unknown error")
        st.error(f"API Error ({response.status_code}): {detail}")
    except ValueError:
        st.error(f"Unexpected API Response ({response.status_code}): {response.text}")


def _fetch_binary(url: str, timeout: float = 40.0) -> Optional[bytes]:
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.content
        else:
            _handle_api_error(r)
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"🔌 Connection Error while fetching {url}: {e}")
        return None


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Empowering Blind Users with an Agentic AI Journalist for Real-Time Global News", page_icon=" 🗞️ ", layout="centered")
    st.title("🗞️ Empowering Blind Users with an Agentic AI Journalist for Real-Time Global News")
    st.markdown("#### 🎙️ News & Reddit Audio/Text Summarizer")

    # Session state
    if "topics" not in st.session_state:
        st.session_state.topics = []
    if "input_key" not in st.session_state:
        st.session_state.input_key = 0

    # Sidebar (settings)
    with st.sidebar:
        st.header("Settings")
        source_type = st.selectbox(
            "Data Sources",
            options=["both", "news", "reddit"],
            index=0,
            format_func=lambda x: {"news": "🌐 News", "reddit": "📑 Reddit", "both": "🔗 Both"}[x],
        )
        return_format_label = st.radio(
            "Return Format",
            options=["Audio", "Text", "Both"],
            index=0,
            help="Choose whether to receive audio, text, or both.",
        )
        return_format: RETURN_FORMAT = {"Audio": "audio", "Text": "json", "Both": "both"}[return_format_label]

    # Topic management
    st.markdown("##### 📝 Topic Management")
    col1, col2 = st.columns([4, 1], vertical_alignment="bottom")
    with col1:
        new_topic = st.text_input(
            "Enter a topic to analyze",
            key=f"topic_input_{st.session_state.input_key}",
            placeholder="e.g., Artificial Intelligence",
            label_visibility="collapsed",
        )
    with col2:
        add_disabled = len(st.session_state.topics) >= 1 or not new_topic.strip()
        if st.button("Add ➕", disabled=add_disabled, use_container_width=True):
            st.session_state.topics.append(new_topic.strip())
            st.session_state.input_key += 1
            st.rerun()

    # Display selected topics
    if st.session_state.topics:
        st.subheader("✅ Selected Topic")
        for i, topic in enumerate(st.session_state.topics[:3]):
            cols = st.columns([6, 1, 1])
            cols[0].write(f"{i + 1}. {topic}")
            if cols[1].button("Rem ❌", key=f"remove_{i}", use_container_width=True):
                del st.session_state.topics[i]
                st.rerun()
            if cols[2].button("Clear 🧹", key="clear_all", use_container_width=True):
                st.session_state.topics = []
                st.rerun()

    st.markdown("---")
    st.subheader("🚀 Generate")

    # Action button
    gen_disabled = len(st.session_state.topics) == 0
    if st.button("Generate Summary", disabled=gen_disabled, type="primary"):
        if not st.session_state.topics:
            st.error("Please add at least one topic.")
            return

        with st.spinner("🔍 Analyzing topics and generating output..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/generate-news-audio",
                    params={"return_format": return_format},
                    json={
                        "topics": st.session_state.topics,
                        "source_type": source_type,
                    },
                    timeout=120,
                )

                # --- Audio-only path
                if return_format == "audio":
                    if resp.status_code == 200:
                        st.audio(resp.content, format="audio/mpeg")
                        st.download_button(
                            "Download Audio Summary",
                            data=resp.content,
                            file_name="news-summary.mp3",
                            type="primary",
                        )
                    else:
                        _handle_api_error(resp)
                    return

                # --- JSON / BOTH path (JSON payload with summary + optional audio_url)
                if resp.status_code == 200:
                    data: Dict[str, Any] = resp.json()

                    # Show text summary
                    st.subheader("📝 Text Summary")
                    st.text_area(
                        "Summary",
                        value=data.get("summary", ""),
                        height=350,
                        key="summary_text_area",
                    )

                    # If BOTH or JSON with audio available
                    if return_format in ("both", "json"):
                        audio_url = data.get("audio_url")
                        if audio_url:
                            full_url = audio_url if not audio_url.startswith("/") else BACKEND_URL.rstrip("/") + audio_url
                            audio_bytes = _fetch_binary(full_url)
                            if audio_bytes:
                                st.subheader("🔊 Audio")
                                st.audio(audio_bytes, format="audio/mpeg")
                                st.download_button(
                                    "Download Audio Summary",
                                    data=audio_bytes,
                                    file_name=os.path.basename(audio_url) if audio_url else "news-summary.mp3",
                                    type="primary",
                                )
                        elif return_format == "both":
                            st.warning("Audio was requested but not produced.")
                else:
                    _handle_api_error(resp)

            except requests.exceptions.ConnectionError:
                st.error("🔌 Connection Error: Could not reach the backend server.")
            except Exception as e:
                st.error(f"⚠️ Unexpected Error: {e}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
