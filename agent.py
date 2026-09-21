import os
import json
import random
import re
from datetime import datetime, timedelta, timezone

import requests
from google import genai
from google.genai import types as genai_types
import anthropic

# ---------- Config ----------
IST = timezone(timedelta(hours=5, minutes=30))
HISTORY_FILE = "history.json"
LINKEDIN_VERSION = "202506"  # LinkedIn API version header (update if LinkedIn rejects it)
LINKEDIN_API = "https://api.linkedin.com"

GEMINI_KEY = os.environ["GEMINI_KEY"]
ANTHROPIC_KEY = os.environ["ANTHROPIC_KEY"]
LI_TOKEN = os.environ["LI_TOKEN"]
TG_TOKEN = os.environ["TG_TOKEN"]
TG_CHAT_ID = os.environ["TG_CHAT_ID"]

gemini_client = genai.Client(api_key=GEMINI_KEY)
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

BIO_FACTS = (
    "B.Tech Biotechnology student at Lovely Professional University (LPU), "
    "interested in healthcare AI, built the MediScan AI (CT scan analyzer) project, "
    "and has worked as a data analyst trainee intern."
)

TOPICS = {
    0: "current developments in AI",
    1: "AI in healthcare",
    2: "biotechnology",
    3: "technology",
    4: "recent discoveries in biology",
    5: "crossover between AI and biotechnology",
}

# ---------- Telegram ----------
def telegram_send(message):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT_ID, "text": message[:4000]},
            timeout=20,
        )
    except Exception as e:
        print(f"Telegram send failed: {e}")

# ---------- History ----------
def load_history():
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def recent_titles(history, limit=15):
    return [h.get("title", "") for h in history[-limit:] if h.get("title")]

def this_week_entries(history):
    now = datetime.now(IST)
    start_of_week = now - timedelta(days=now.weekday())
    start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
    result = []
    for h in history:
        try:
            entry_date = datetime.fromisoformat(h["date"])
        except Exception:
            continue
        if entry_date >= start_of_week:
            result.append(h)
    return result

# ---------- Research (Gemini + Google Search grounding) ----------
def research_topic(topic, avoid_titles):
    avoid_text = ""
    if avoid_titles:
        avoid_text = "Do NOT repeat these already-covered stories/angles: " + "; ".join(avoid_titles) + ".\n"

    prompt = (
        f"Research a specific, recent, interesting story or development in the area of: {topic}.\n"
        f"{avoid_text}"
        "Respond in exactly this format:\n"
        "TITLE: <a short unique title for this specific story, under 10 words>\n"
        "RESEARCH:\n<3-6 factual bullet points about this story, based on real, current information>"
    )

    response = gemini_client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
        )
    text = response.text or ""

    title_match = re.search(r"TITLE:\s*(.+)", text)
    research_match = re.search(r"RESEARCH:\s*(.+)", text, re.DOTALL)

    title = title_match.group(1).strip() if title_match else topic
    research = research_match.group(1).strip() if research_match else text

    if not research:
        return None, None
    return title, research

# ---------- Writing the post (Claude) ----------
def write_post(topic, title, research):
    prompt = (
        f"Write a LinkedIn post (150-200 words) in first person, simple English, "
        f"about this topic: {title} (category: {topic}).\n\n"
        f"Base it ONLY on these researched facts:\n{research}\n\n"
        f"You may mention these facts about the author ONLY if naturally relevant, and do not invent anything beyond them:\n{BIO_FACTS}\n\n"
        "Do not invent any personal experience, story, or achievement not listed above. "
        "Make it sound human and genuine, not like an ad. No hashtags overload — at most 3 relevant hashtags at the end. "
        "Return ONLY the post text, nothing else."
    )
    response = claude_client.messages.create(
        model="claude-sonnet-5",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()

def write_recap_post(entries):
    posts_summary = "\n\n".join(f"- {e.get('topic')}: {e.get('title')}" for e in entries)
    prompt = (
        "Write a short LinkedIn weekly recap post (150-200 words), first person, simple English, "
        "summarizing ONLY the following topics I posted about this week. Do not add any new facts, "
        "just reflect on and connect these:\n\n" + posts_summary + "\n\n"
        "Return ONLY the post text, nothing else."
    )
    response = claude_client.messages.create(
        model="claude-sonnet-5",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()

# ---------- Fact-check safety gate (Claude) ----------
def fact_check(post_text, research):
    prompt = (
        f"Research notes:\n{research}\n\n"
        f"LinkedIn post draft:\n{post_text}\n\n"
        "Check if every factual claim in the post draft is supported by the research notes above. "
        "Reply with exactly 'OK' if fully supported, or 'FAIL: <short reason>' if any claim is not supported."
    )
    response = claude_client.messages.create(
        model="claude-sonnet-5",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()

# ---------- Image generation (Gemini) ----------
def generate_image(title):
    prompt = (
        f"A clean, simple explainer-style diagram illustrating: {title}. "
        "Flat design, minimal text, professional, suitable for a LinkedIn post."
    )
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-image-preview",
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
            ),
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                return part.inline_data.data
    except Exception as e:
        print(f"Image generation failed: {e}")
    return None

# ---------- LinkedIn ----------
def li_headers():
    return {
        "Authorization": f"Bearer {LI_TOKEN}",
        "X-Restli-Protocol-Version": "2.0.0",
        "LinkedIn-Version": LINKEDIN_VERSION,
        "Content-Type": "application/json",
    }

def linkedin_get_owner_urn():
    resp = requests.get(
        f"{LINKEDIN_API}/v2/userinfo",
        headers={"Authorization": f"Bearer {LI_TOKEN}"},
        timeout=20,
    )
    resp.raise_for_status()
    sub = resp.json()["sub"]
    return f"urn:li:person:{sub}"

def linkedin_upload_image(owner_urn, image_bytes):
    init = requests.post(
        f"{LINKEDIN_API}/rest/images?action=initializeUpload",
        headers=li_headers(),
        json={"initializeUploadRequest": {"owner": owner_urn}},
        timeout=30,
    )
    init.raise_for_status()
    value = init.json()["value"]
    upload_url = value["uploadUrl"]
    image_urn = value["image"]

    put = requests.put(
        upload_url,
        headers={"Authorization": f"Bearer {LI_TOKEN}"},
        data=image_bytes,
        timeout=60,
    )
    put.raise_for_status()
    return image_urn

def escape_linkedin_text(text):
    reserved = ['\\', '|', '{', '}', '@', '[', ']', '(', ')', '<', '>', '*', '_', '~']
    out = []
    for ch in text:
        if ch in reserved:
            out.append('\\' + ch)
        else:
            out.append(ch)
    return "".join(out)

def linkedin_create_post(owner_urn, commentary, image_urn):
    body = {
        "author": owner_urn,
        "commentary": commentary,
        "visibility": "PUBLIC",
        "distribution": {"feedDistribution": "MAIN_FEED"},
        "content": {"media": {"id": image_urn}},
        "lifecycleState": "PUBLISHED",
        "isReshareDisabledByAuthor": False,
    }
    resp = requests.post(
        f"{LINKEDIN_API}/rest/posts",
        headers=li_headers(),
        json=body,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.headers.get("x-restli-id", "unknown")

# ---------- Main ----------
def main():
    now = datetime.now(IST)
    weekday = now.weekday()  # Monday=0 ... Sunday=6
    history = load_history()

    if weekday == 6:
        entries = this_week_entries(history)
        if not entries:
            telegram_send("Sunday recap skipped: no posts found from this week in history.json.")
            return
        title = "Weekly recap"
        topic = "weekly recap"
        post_text = write_recap_post(entries)
    else:
        topic = TOPICS[weekday]
        avoid = recent_titles(history)
        title, research = research_topic(topic, avoid)
        if not research:
            telegram_send(f"Post skipped: research step returned nothing for topic '{topic}'.")
            return

        post_text = write_post(topic, title, research)
        check = fact_check(post_text, research)
        if check != "OK":
            telegram_send(f"Post SKIPPED (fact-check failed) for topic '{topic}' / '{title}': {check}")
            return

    image_bytes = generate_image(title)
    if not image_bytes:
        telegram_send(f"Post skipped: image generation failed for '{title}'. No text-only post was made.")
        return

    final_text = escape_linkedin_text(post_text) + "\n\nDiagram: AI-generated"

    try:
        owner_urn = linkedin_get_owner_urn()
        image_urn = linkedin_upload_image(owner_urn, image_bytes)
        post_id = linkedin_create_post(owner_urn, final_text, image_urn)
    except Exception as e:
        telegram_send(f"Post FAILED at LinkedIn API step for '{title}': {e}")
        raise

    history.append({
        "date": now.isoformat(),
        "weekday": weekday,
        "topic": topic,
        "title": title,
        "post": post_text,
        "post_id": post_id,
    })
    save_history(history)

    telegram_send(f"✅ Posted today's LinkedIn update: '{title}' (topic: {topic}). Post ID: {post_id}")

if __name__ == "__main__":
    main()
