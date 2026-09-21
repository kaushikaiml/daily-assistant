"""
Daily LinkedIn agent (free version, Gemini only).

What it does every day:
 1. Finds fresh news (last 7 days) from RSS feeds for today's topic
 2. Picks one story it has not used before
 3. Reads the article and writes a human-sounding LinkedIn post (Gemini)
 4. Fact-checks the post against the article. If not supported -> SKIP + Telegram alert
 5. Makes a clean summary image (drawn with code, no paid image API needed)
 6. Posts to LinkedIn (official API) or, in PREVIEW mode, sends the draft to Telegram
 7. Saves the record in history.json

Secrets needed (GitHub -> Settings -> Secrets): GEMINI_KEY, LI_TOKEN, TG_TOKEN, TG_CHAT_ID
"""

import datetime as dt
import html
import io
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from urllib.parse import quote, urlparse

import requests

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # image card will be skipped if Pillow is missing
    Image = None

# =====================================================================
# SETTINGS  (you can edit these lines on GitHub)
# =====================================================================
PREVIEW_ONLY = True   # True  = send the draft to Telegram ONLY (nothing goes to LinkedIn)
                      # False = post live on LinkedIn.  Change to False after you like the previews.
USE_AI_IMAGE = False  # False = draw a clean summary card with code (free, always works)
                      # True  = try Gemini image models first (usually needs a paid plan)

TEXT_MODELS = ["gemini-3.1-flash-lite", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
IMAGE_MODELS = ["gemini-3.1-flash-lite-image", "gemini-2.5-flash-image"]
MAX_AGE_DAYS = 7
HISTORY_FILE = "history.json"
IMAGE_DIR = "images"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))

# Only REAL facts about you. The AI is not allowed to invent anything else.
ABOUT_ME = (
    "Mohit, B.Tech Biotechnology student at LPU. Interested in healthcare AI. "
    "Built MediScan AI (an AI cancer detection app). Did a data analyst trainee internship."
)


def gnews(query):
    """Google News RSS for a search query, last 7 days."""
    return ("https://news.google.com/rss/search?q=" + quote(query + " when:7d")
            + "&hl=en-IN&gl=IN&ceid=IN:en")


# weekday number (Monday = 0) -> (topic name, list of RSS feeds)
TOPICS = {
    0: ("Current AI", [
        "https://techcrunch.com/category/artificial-intelligence/feed/",
        "https://www.technologyreview.com/feed/",
        gnews("artificial intelligence"),
    ]),
    1: ("AI in healthcare", [
        "https://www.statnews.com/feed/",
        gnews("AI healthcare hospital"),
        gnews("medical AI FDA"),
    ]),
    2: ("Biotechnology", [
        gnews("biotechnology"),
        gnews("gene therapy OR CRISPR"),
        "https://www.statnews.com/feed/",
    ]),
    3: ("Technology", [
        "https://www.theverge.com/rss/index.xml",
        "https://feeds.arstechnica.com/arstechnica/index",
        gnews("technology"),
    ]),
    4: ("Recent discoveries in biology", [
        "https://www.nature.com/nature.rss",
        "https://phys.org/rss-feed/biology-news/",
        gnews("new biology discovery scientists"),
    ]),
    5: ("AI meets biotech", [
        gnews("AI drug discovery"),
        gnews("AI protein biology"),
        "https://www.nature.com/nature.rss",
    ]),
    6: ("Weekly recap", []),  # built from this week's own posts
}

UA = {"User-Agent": "Mozilla/5.0 (compatible; DailyAssistant/1.0)"}


class Skip(Exception):
    """Raised when we deliberately do not post (with the reason)."""


# =====================================================================
# Small helpers
# =====================================================================
def log(*a):
    print(*a, flush=True)


def telegram_text(msg):
    token, chat = os.environ.get("TG_TOKEN"), os.environ.get("TG_CHAT_ID")
    if not token or not chat:
        log("[telegram not set] " + msg)
        return
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat, "text": msg[:4000],
                            "disable_web_page_preview": "true"}, timeout=30)
    except Exception as e:
        log("Telegram error:", e)


def telegram_photo(img_bytes, caption=""):
    token, chat = os.environ.get("TG_TOKEN"), os.environ.get("TG_CHAT_ID")
    if not token or not chat or not img_bytes:
        return
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendPhoto",
                      data={"chat_id": chat, "caption": caption[:1000]},
                      files={"photo": ("post.png", img_bytes)}, timeout=60)
    except Exception as e:
        log("Telegram photo error:", e)


def load_history():
    try:
        with open(HISTORY_FILE, encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def strip_html(text):
    text = re.sub(r"<(script|style).*?</\1>", " ", text or "", flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


# =====================================================================
# Gemini (with retries and backup models)
# =====================================================================
_client = None


def gemini_client():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=os.environ["GEMINI_KEY"])
    return _client


def gemini(prompt, json_mode=False):
    from google.genai import types
    last = None
    for model in TEXT_MODELS:
        for attempt in range(3):
            try:
                cfg = types.GenerateContentConfig(
                    temperature=0.6,
                    response_mime_type="application/json" if json_mode else "text/plain")
                r = gemini_client().models.generate_content(model=model, contents=prompt, config=cfg)
                if r.text:
                    return r.text
            except Exception as e:
                last = e
                msg = str(e)
                if "404" in msg or "NOT_FOUND" in msg:
                    break  # this model name is not available -> try the next one
                wait = 20 * (attempt + 1)  # 503 busy / 429 limit -> wait and retry
                log(f"Gemini {model} problem (attempt {attempt + 1}): {msg[:120]} ... waiting {wait}s")
                time.sleep(wait)
    raise Exception(f"Gemini failed on all models: {str(last)[:200]}")


def parse_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    return json.loads(text)


# =====================================================================
# Step 1: news from RSS
# =====================================================================
def local(tag):
    return tag.split("}")[-1].lower()


def parse_date(s):
    if not s:
        return None
    s = s.strip()
    try:
        d = parsedate_to_datetime(s)
    except Exception:
        try:
            d = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d


def parse_feed(xml_bytes, feed_url):
    items = []
    root = ET.fromstring(xml_bytes)
    feed_host = urlparse(feed_url).netloc.replace("www.", "")
    for el in root.iter():
        if local(el.tag) not in ("item", "entry"):
            continue
        f = {"title": "", "link": "", "date": None, "summary": ""}
        for c in el:
            name = local(c.tag)
            if name == "title":
                f["title"] = strip_html(c.text)
            elif name == "link":
                f["link"] = (c.text or c.attrib.get("href", "")).strip()
            elif name in ("pubdate", "date", "published", "updated") and not f["date"]:
                f["date"] = parse_date(c.text)
            elif name in ("description", "summary", "content") and not f["summary"]:
                f["summary"] = strip_html(c.text)[:600]
        if f["title"] and f["link"] and f["date"]:
            # Google News titles end with " - Publisher"
            if "news.google.com" in feed_host and " - " in f["title"]:
                f["source"] = f["title"].rsplit(" - ", 1)[1]
                f["title"] = f["title"].rsplit(" - ", 1)[0]
            else:
                f["source"] = feed_host
            items.append(f)
    return items


def get_candidates(feeds, used_links, used_titles):
    now = dt.datetime.now(dt.timezone.utc)
    out, seen = [], set()
    for url in feeds:
        try:
            r = requests.get(url, headers=UA, timeout=25)
            r.raise_for_status()
            found = parse_feed(r.content, url)
            log(f"Feed OK: {url[:70]} -> {len(found)} items")
        except Exception as e:
            log(f"Feed failed: {url[:70]} -> {str(e)[:80]}")
            continue
        for it in found:
            age = (now - it["date"]).total_seconds() / 86400
            key = it["title"].lower()
            if age < 0 or age > MAX_AGE_DAYS or key in seen:
                continue
            if it["link"] in used_links or key in used_titles:
                continue
            seen.add(key)
            it["age_days"] = round(age, 1)
            out.append(it)
    out.sort(key=lambda x: x["date"], reverse=True)
    return out[:14]


def read_article(item):
    """Try to read the real article text. Falls back to the RSS summary."""
    link = item["link"]
    text, final = "", link
    if "news.google.com" not in link:
        try:
            r = requests.get(link, headers=UA, timeout=25)
            final = r.url
            paras = re.findall(r"<p[^>]*>(.*?)</p>", r.text, flags=re.S | re.I)
            paras = [strip_html(p) for p in paras]
            text = " ".join(p for p in paras if len(p) > 60)[:5000]
        except Exception as e:
            log("Could not read article:", str(e)[:80])
    if len(text) < 300:
        text = f"{item['title']}. {item['summary']}"
    return final, text


# =====================================================================
# Step 2: write + fact-check
# =====================================================================
STYLE_RULES = f"""
Write a LinkedIn post for this author: {ABOUT_ME}
Rules:
- First person, simple English, warm and natural, like a student sharing what he learned. Not salesy.
- Medium length: 150 to 190 words. Short paragraphs. Strong first line (a hook, not a headline copy).
- Use "→" at the start of 2 or 3 key points.
- Use ONLY facts that appear in the SOURCE TEXT. No invented numbers, quotes, names or dates.
- Never invent personal experiences for the author. Only use the author facts above, and only if it fits naturally.
- Give one honest takeaway or opinion, then end with a genuine question to the reader.
- No buzzwords like "game-changer", no more than 1 emoji.
- Do NOT include hashtags or a source line (they are added later).
"""


def write_post(topic, item, article_text, today, problems=None):
    fix = ""
    if problems:
        fix = "\nA fact-checker found these problems in your previous draft. Fix them:\n- " + "\n- ".join(problems)
    prompt = f"""{STYLE_RULES}
Today's date: {today.strftime('%d %B %Y')}. Story published: {item['date'].strftime('%d %B %Y')} ({item['age_days']} days ago).
Only use words like "today", "just", "this week" if the story is 2 days old or less; otherwise say the month or "recently".
Topic of the day: {topic}
STORY TITLE: {item['title']}
PUBLISHER: {item['source']}
SOURCE TEXT:
{article_text}
{fix}
Return JSON only:
{{"post": "the post text", "hashtags": ["#AI", "..."] (3 to 5), "headline": "max 9 words for an image card",
 "points": ["3 short takeaways, each max 12 words"], "publisher": "publisher name"}}"""
    return parse_json(gemini(prompt, json_mode=True))


def fact_check(post, source_text):
    prompt = f"""You are a strict fact-checker.
SOURCE TEXT:
{source_text}

POST:
{post}

Check every factual claim about the story (numbers, names, dates, who did what) against the SOURCE TEXT.
Opinions and questions are fine. Statements about the author are fine only if they match: {ABOUT_ME}
Return JSON only: {{"ok": true or false, "problems": ["each unsupported claim, short"]}}"""
    return parse_json(gemini(prompt, json_mode=True))


def write_recap(history, today):
    week = [h for h in history if h.get("status") == "posted"
            and h.get("date", "") >= (today - dt.timedelta(days=7)).strftime("%Y-%m-%d")]
    if len(week) < 2:
        return None, None
    source = "\n\n".join(f"Post on {h['date']} ({h.get('topic', '')}): {h.get('post', '')}" for h in week)
    prompt = f"""{STYLE_RULES}
This is a WEEKLY RECAP post. Use ONLY the posts below (from this week). Add no new facts.
Mention what the week's posts covered and one common theme. End with a question.
POSTS FROM THIS WEEK:
{source}
Return JSON only:
{{"post": "the recap text", "hashtags": ["#AI", "..."] (3 to 5), "headline": "max 9 words",
 "points": ["3 short takeaways, each max 12 words"], "publisher": "my posts this week"}}"""
    return parse_json(gemini(prompt, json_mode=True)), source


# =====================================================================
# Step 3: image
# =====================================================================
ACCENTS = ["#0F766E", "#3B4CCA", "#B4433A", "#7A4FB5", "#1D6FA5", "#2F7D32", "#C77700"]


def _font(size, bold=False):
    paths = ([
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ] if bold else [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ])
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _wrap(draw, text, font, max_w, max_lines):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= max_w:
            cur = t
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(" .,") + "..."
    return lines


def make_card(headline, points, publisher, weekday):
    """Clean 1200x627 summary card drawn with code. Free and never fails on quota."""
    if Image is None:
        return None
    S = 2
    W, H = 1200 * S, 627 * S
    accent = ACCENTS[weekday % len(ACCENTS)]
    img = Image.new("RGB", (W, H), "#F6F7F9")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 14 * S, H], fill=accent)

    y = 52 * S
    title_font = _font(50 * S, bold=True)
    for line in _wrap(d, headline, title_font, 1060 * S, 3):
        d.text((64 * S, y), line, font=title_font, fill="#1B2430")
        y += 62 * S
    y += 26 * S

    pf = _font(28 * S)
    for p in (points or [])[:3]:
        d.ellipse([64 * S, y + 11 * S, 78 * S, y + 25 * S], fill=accent)
        for line in _wrap(d, p, pf, 1010 * S, 2):
            d.text((96 * S, y), line, font=pf, fill="#1B2430")
            y += 38 * S
        y += 20 * S

    ff = _font(18 * S)
    d.text((64 * S, 583 * S), f"Source: {publisher}"[:70], font=ff, fill="#5B6675")
    right = "Summary card made with AI"
    d.text((W - 64 * S - d.textlength(right, font=ff), 583 * S), right, font=ff, fill="#5B6675")

    img = img.resize((1200, 627), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def ai_image(headline):
    from google.genai import types
    prompt = ("Clean, modern, professional editorial illustration for a LinkedIn post about: "
              f"{headline}. No text, no letters, no logos, no real people.")
    for model in IMAGE_MODELS:
        try:
            r = gemini_client().models.generate_content(
                model=model, contents=prompt,
                config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]))
            for part in r.candidates[0].content.parts:
                if getattr(part, "inline_data", None) and part.inline_data.data:
                    return part.inline_data.data
        except Exception as e:
            log(f"AI image {model} failed: {str(e)[:120]}")
    return None


# =====================================================================
# Step 4: LinkedIn (official API)
# =====================================================================
def li_headers():
    # LinkedIn only accepts recent API versions. Use "2 months ago" so it is always valid.
    version = (dt.datetime.now() - dt.timedelta(days=60)).strftime("%Y%m")
    return {
        "Authorization": f"Bearer {os.environ['LI_TOKEN']}",
        "LinkedIn-Version": version,
        "X-Restli-Protocol-Version": "2.0.0",
        "Content-Type": "application/json",
    }


def li_escape(text):
    for ch in "\\|{}@[]()<>*_~":
        text = text.replace(ch, "\\" + ch)
    return text  # '#' is NOT escaped so hashtags still work


def post_to_linkedin(text, img_bytes, alt):
    h = li_headers()
    me = requests.get("https://api.linkedin.com/v2/userinfo",
                      headers={"Authorization": h["Authorization"]}, timeout=30)
    if me.status_code == 401:
        raise Exception("LinkedIn token expired or invalid. Make a new token and update the LI_TOKEN secret.")
    me.raise_for_status()
    urn = f"urn:li:person:{me.json()['sub']}"

    content = None
    if img_bytes:
        init = requests.post("https://api.linkedin.com/rest/images?action=initializeUpload",
                             headers=h, json={"initializeUploadRequest": {"owner": urn}}, timeout=30)
        init.raise_for_status()
        val = init.json()["value"]
        up = requests.put(val["uploadUrl"], data=img_bytes,
                          headers={"Authorization": h["Authorization"]}, timeout=60)
        up.raise_for_status()
        content = {"media": {"title": alt[:100] or "Image", "altText": alt[:300], "id": val["image"]}}

    body = {
        "author": urn,
        "commentary": li_escape(text),
        "visibility": "PUBLIC",
        "distribution": {"feedDistribution": "MAIN_FEED", "targetEntities": [],
                         "thirdPartyDistributionChannels": []},
        "lifecycleState": "PUBLISHED",
        "isReshareDisabledByAuthor": False,
    }
    if content:
        body["content"] = content
    r = requests.post("https://api.linkedin.com/rest/posts", headers=h, json=body, timeout=60)
    if r.status_code not in (200, 201):
        raise Exception(f"LinkedIn post failed {r.status_code}: {r.text[:300]}")
    post_id = r.headers.get("x-restli-id", "")
    return f"https://www.linkedin.com/feed/update/{post_id}" if post_id else ""


# =====================================================================
# Main
# =====================================================================
def run(history, today, entry):
    weekday = today.weekday()
    topic, feeds = TOPICS[weekday]
    entry["topic"] = topic
    used_links = {h.get("source_url") for h in history if h.get("status") == "posted"}
    used_titles = {h.get("title", "").lower() for h in history if h.get("status") == "posted"}

    data, source_text, item = None, "", None

    if weekday == 6:  # Sunday recap
        data, source_text = write_recap(history, today)
        if data is None:  # not enough posts yet -> use a normal biology story
            topic, feeds = TOPICS[4]
            entry["topic"] = topic
    if data is None:
        cands = get_candidates(feeds, used_links, used_titles)
        if not cands:
            raise Skip("No fresh (last 7 days) unused stories found in the feeds.")
        menu = "\n".join(f"{i}. [{c['age_days']}d old] {c['title']} ({c['source']})"
                         for i, c in enumerate(cands))
        pick = parse_json(gemini(
            f"Topic today: {topic}.\nPick the ONE story that is most interesting and relevant "
            f"for a biotech student who follows healthcare AI. Prefer serious sources, avoid rumours.\n"
            f"{menu}\nReturn JSON only: {{\"pick\": number}}", json_mode=True))
        item = cands[int(pick["pick"])]
        final_url, article = read_article(item)
        source_text = article
        entry["title"] = item["title"]
        entry["source_url"] = item["link"]
        log("Chosen story:", item["title"], "|", item["source"], f"| {item['age_days']} days old")
        data = write_post(topic, item, article, today)

    # fact-check (one rewrite allowed)
    check = fact_check(data["post"], source_text)
    if not check.get("ok"):
        log("Fact-check problems:", check.get("problems"))
        if item is not None:
            data = write_post(topic, item, source_text, today, problems=check.get("problems"))
            check = fact_check(data["post"], source_text)
        if not check.get("ok"):
            raise Skip("Fact-check failed: " + "; ".join(check.get("problems", []))[:400])

    # build the final text
    tags = " ".join(t if t.startswith("#") else "#" + t for t in data.get("hashtags", [])[:5])
    publisher = data.get("publisher") or (item["source"] if item else "")
    if item is not None and "news.google.com" not in item["link"]:
        source_line = f"Source: {publisher} - {item['link']}"
    elif item is not None:
        source_line = f"Source: {publisher}"
    else:
        source_line = ""
    final_text = "\n\n".join(x for x in [data["post"].strip(), source_line, tags] if x)
    entry["post"] = final_text

    # image
    img = None
    if USE_AI_IMAGE:
        img = ai_image(data.get("headline", topic))
    if img is None:
        img = make_card(data.get("headline", topic), data.get("points", []), publisher, weekday)
    if img is None:
        telegram_text("Pillow is missing, posting without an image. Add 'pillow' to requirements.txt")
    else:
        os.makedirs(IMAGE_DIR, exist_ok=True)
        path = f"{IMAGE_DIR}/{today.strftime('%Y-%m-%d')}.png"
        with open(path, "wb") as f:
            f.write(img)
        entry["image"] = path

    if PREVIEW_ONLY:
        entry["status"] = "preview"
        telegram_text("PREVIEW (nothing posted to LinkedIn)\nTopic: " + topic + "\n\n" + final_text)
        telegram_photo(img, "Image for the preview post")
        log("PREVIEW sent to Telegram.")
        return

    link = post_to_linkedin(final_text, img, data.get("headline", topic))
    entry["status"] = "posted"
    entry["linkedin_url"] = link
    telegram_text(f"Posted on LinkedIn today ({topic}).\n{link}")
    log("Posted:", link)


def main():
    today = dt.datetime.now(IST)
    history = load_history()
    entry = {"date": today.strftime("%Y-%m-%d"), "time": today.strftime("%H:%M"),
             "topic": "", "status": "", "title": "", "source_url": "", "post": "",
             "image": "", "linkedin_url": "", "reason": ""}
    exit_code = 0
    try:
        run(history, today, entry)
    except Skip as e:
        entry["status"], entry["reason"] = "skipped", str(e)
        telegram_text(f"Post SKIPPED today.\nReason: {e}")
        log("SKIPPED:", e)
        exit_code = 1
    except Exception as e:
        entry["status"], entry["reason"] = "failed", str(e)[:400]
        telegram_text(f"Robot FAILED today.\nReason: {str(e)[:500]}")
        log("FAILED:", e)
        exit_code = 1
    history.append(entry)
    save_history(history)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
        
