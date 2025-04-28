import os
import requests
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from google import genai  # unified Gen AI SDK

# ——— Setup ———
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("Set GOOGLE_API_KEY in your .env")
    st.stop()

TEAMS_WEBHOOK_URL = os.getenv("TEAMS_WEBHOOK_URL")
if not TEAMS_WEBHOOK_URL:
    st.error("Set TEAMS_WEBHOOK_URL in your .env")
    st.stop()

# Initialize the GenAI client
client = genai.Client(api_key=API_KEY)

# Prompt header for combined summarization
PROMPT_HEADER = """You are a professional video summarizer. You will be provided with multiple video transcripts.
Summarize the overall combined insights from all these videos together in clear bullet points within 350 words.
Include:
- Key highlights from each video
- Common themes/topics discussed
- Speaker-specific important notes if available
- Any contrasts or different viewpoints between videos
- A final 2-line overall takeaway conclusion.

Here are the combined transcripts: 
"""

# ——— Transcript Extraction ———
def extract_transcript(youtube_url: str) -> str:
    """Pulls full text transcript from a YouTube URL."""
    video_id = youtube_url.split("v=")[-1]
    segments = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join(seg["text"] for seg in segments)

def fetch_transcript(source: str) -> str:
    """Handles YouTube or any HTTP-served transcript text."""
    if "youtube.com" in source:
        return extract_transcript(source)
    resp = requests.get(source)
    resp.raise_for_status()
    return resp.text

# ——— Summarization ———
def generate_summary(combined_transcript: str) -> str:
    """Calls the Gemini text model to summarize."""
    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[PROMPT_HEADER + combined_transcript]
    )
    return resp.text

# ——— Embedding ———
def generate_embedding(transcript: str) -> list[float]:
    """Calls the Gemini embedding model to vectorize the transcript."""
    resp = client.models.embed_content(
        model="gemini-embedding-exp-03-07",
        contents=transcript
    )
    return resp.embeddings

# ——— Teams Integration ———
def post_to_teams(summary_text: str):
    """Posts summary text to a Microsoft Teams channel via webhook."""
    payload = {"text": f"**📽️ Combined Video Summary**\n\n{summary_text}"}
    resp = requests.post(TEAMS_WEBHOOK_URL, json=payload)
    if resp.status_code != 200:
        st.error(f"Teams webhook failed: {resp.text}")

# ——— Chatbot for Q&A ———
def ask_chatbot(summary_text: str, question: str) -> str:
    """Generates an answer from Gemini based on the summary and user question."""
    prompt = f"""You are an assistant. Based on the following video summary, answer the user’s question clearly and briefly.

Video Summary:
{summary_text}

User Question:
{question}

Answer:"""
    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt]
    )
    return resp.text

# ——— Streamlit UI ———
st.title("Multi-Video Transcript to Combined Summary & Q&A")

urls_input = st.text_area(
    "Enter video links or transcript URLs (comma-separated):",
    height=100
)

post_to_teams_checkbox = st.checkbox("Post summary to Microsoft Teams", value=True)
enable_chatbot = st.checkbox("Enable Chatbot Q&A after summary", value=False)

if st.button("Generate Combined Summary"):
    if not urls_input:
        st.warning("Please enter at least one URL.")
    else:
        urls = [u.strip() for u in urls_input.split(",") if u.strip()]
        combined_transcript = ""
        with st.spinner(f"Fetching transcripts for {len(urls)} video(s)..."):
            for idx, url in enumerate(urls, 1):
                st.write(f"Fetching transcript {idx} of {len(urls)}")
                try:
                    combined_transcript += f"\n\n=== Transcript {idx} ===\n" + fetch_transcript(url)
                except Exception as e:
                    st.error(f"Error fetching transcript for {url}: {e}")
                    combined_transcript += ""
        with st.spinner("Generating combined summary..."):
            try:
                summary = generate_summary(combined_transcript)
                st.subheader("Combined Detailed Notes")
                st.write(summary)
                st.success("Summary generated successfully!")

                # Post to Teams
                if post_to_teams_checkbox:
                    post_to_teams(summary)
                    st.success("Summary posted to Microsoft Teams!")

                # Show embedding dims
                embedding = generate_embedding(combined_transcript)
                st.write("Embedding (first 5 dims):", embedding[:5])

                # Store summary for chatbot
                st.session_state.summary = summary
            except Exception as e:
                st.error(f"Error generating summary: {e}")

# Chatbot interface
if enable_chatbot and st.session_state.get("summary"):
    st.markdown("---")
    st.subheader("Chatbot Q&A")
    question = st.text_input("Ask a question about the summary:", key="chat_question")
    if st.button("Ask Bot", key="ask_bot") and question:
        with st.spinner("Bot is thinking..."):
            try:
                answer = ask_chatbot(st.session_state.summary, question)
                st.write("**Answer:**", answer)
            except Exception as e:
                st.error(f"Error in chatbot response: {e}")
