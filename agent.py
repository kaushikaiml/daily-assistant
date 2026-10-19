"""
Daily LinkedIn Agent
====================

An automated agent that researches a topic, writes a LinkedIn post with
Gemini, verifies it, and publishes it once a day.

Daily workflow:
    1. Collect recent news (last 7 days) from RSS feeds for the day's topic.
    2. Select one story that has not been used before.
    3. Read the article and generate a LinkedIn post with Gemini.
    4. Fact-check the post against the article. If any claim is not
       supported, the post is skipped and an alert is sent via Telegram.
    5. Generate a summary image programmatically (no paid image API).
    6. Publish through the official LinkedIn API, or, in preview mode,
       send the draft to Telegram instead.
    7. Record the result in history.json.

Required secrets (GitHub: Settings > Secrets and variables > Actions):
    GEMINI_KEY, LI_TOKEN, TG_TOKEN, TG_CHAT_ID
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
except ImportError:
    # Pillow is optional at import time; the summary image is skipped without it.
    Image = None


# =====================================================================
# CONFIGURATION
# =====================================================================
# Publishing mode.
#   True  - send the draft to Telegram only; nothing is posted to LinkedIn.
#   False - publish the post live on LinkedIn.
PREVIEW_ONLY = False

# Image source.
#   False - draw the summary card programmatically (free and reliable).
#   True  - try Gemini image models first (usually requires a paid plan).
USE_AI_IMAGE = False

# Gemini models, in order of preference. The next model is used as a
# fallback when the previous one is unavailable.
TEXT_MODELS = ["gemini-3.1-flash-lite", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
IMAGE_MODELS = ["gemini-3.1-flash-lite-image", "gemini-2.5-flash-image"]

MAX_AGE_DAYS = 7        # Maximum age of a story, in days
MAX_CANDIDATES = 14     # Maximum number of stories offered to Gemini
HISTORY_FILE = "history.json"
IMAGE_DIR = "images"

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DailyAssistant/1.0)"}

# Verified facts about the author. The model must not add any others.
ABOUT_ME = (
    "Mohit, B.Tech Biotechnology student at LPU. Interested in healthcare AI. "
    "Built MediScan AI (an AI cancer detection app). Did a data analyst trainee internship."
)


# =====================================================================
# TOPIC SCHEDULE
# Maps the weekday number (Monday = 0) to (topic name, RSS feed URLs).
# =====================================================================
def google_news(query):
    """Return a Google News RSS URL for the given query (last 7 days)."""
    return (
        "https://news.google.com/rss/search?q=" + quote(query + " when:7d")
        + "&hl=en-IN&gl=IN&ceid=IN:en"
    )


TOPICS = {
    0: ("Current AI", [
        "https://techcrunch.com/category/artificial-intelligence/feed/",
        "https://www.technologyreview.com/feed/",
        google_news("artificial intelligence"),
    ]),
    1: ("AI in healthcare", [
        "https://www.statnews.com/feed/",
        google_news("AI healthcare hospital"),
        google_news("medical AI FDA"),
    ]),
    2: ("Biotechnology", [
        google_news("biotechnology"),
        google_news("gene therapy OR CRISPR"),
        "https://www.statnews.com/feed/",
    ]),
    3: ("Technology", [
        "https://www.theverge.com/rss/index.xml",
        "https://feeds.arstechnica.com/arstechnica/index",
        google_news("technology"),
    ]),
    4: ("Recent discoveries in biology", [
        "https://www.nature.com/nature.rss",
        "https://phys.org/rss-feed/biology-news/",
        google_news("new biology discovery scientists"),
    ]),
    5: ("AI meets biotech", [
        google_news("AI drug discovery"),
        google_news("AI protein biology"),
        "https://www.nature.com/nature.rss",
    ]),
    6: ("Weekly recap", []),  # Generated from the week's own posts, not from feeds
}

# Fallback topic for Sunday when there are too few posts to summarize.
BIOLOGY_TOPIC_DAY = 4


class Skip(Exception):
    """Raised when a post is intentionally not published; the message states why."""


# =====================================================================
# GENERAL UTILITIES
# =====================================================================
def log(*args):
    """Print a message immediately so it appears in the GitHub Actions log."""
    print(*args, flush=True)


def strip_html(text):
    """Remove HTML tags and redundant whitespace from the given text."""
    text = re.sub(r"<(script|style).*?</\1>", " ", text or "", flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def load_history():
    """Load the post history; return an empty list if the file is missing or invalid."""
    try:
        with open(HISTORY_FILE, encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_history(history):
    """Persist the post history to disk."""
    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)


# =====================================================================
# TELEGRAM NOTIFICATIONS
# =====================================================================
def _telegram_credentials():
    """Return (bot token, chat id) from the environment, or (None, None) if unset."""
    return os.environ.get("TG_TOKEN"), os.environ.get("TG_CHAT_ID")


def telegram_text(message):
    """Send a text message to Telegram. Failures are logged and never interrupt the run."""
    token, chat_id = _telegram_credentials()
    if not token or not chat_id:
        log("[telegram not configured] " + message)
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={
                "chat_id": chat_id,
                "text": message[:4000],  # Telegram limit is 4096 characters
                "disable_web_page_preview": "true",
            },
            timeout=30,
        )
    except Exception as error:
        log("Telegram error:", error)


def telegram_photo(image_bytes, caption=""):
    """Send an image to Telegram. Failures are logged and never interrupt the run."""
    token, chat_id = _telegram_credentials()
    if not token or not chat_id or not image_bytes:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendPhoto",
            data={"chat_id": chat_id, "caption": caption[:1000]},
            files={"photo": ("post.png", image_bytes)},
            timeout=60,
        )
    except Exception as error:
        log("Telegram photo error:", error)


# =====================================================================
# GEMINI CLIENT
# =====================================================================
_gemini_client = None


def get_gemini_client():
    """Return a shared Gemini client, creating it on first use."""
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=os.environ["GEMINI_KEY"])
    return _gemini_client


def ask_gemini(prompt, json_mode=False):
    """
    Send a prompt to Gemini and return the response text.

    Each model is retried up to three times with an increasing delay to
    handle temporary overload (503) and rate limits (429). If a model is
    not found (404), the next model in TEXT_MODELS is used.
    """
    from google.genai import types

    last_error = None
    for model in TEXT_MODELS:
        for attempt in range(3):
            try:
                config = types.GenerateContentConfig(
                    temperature=0.6,
                    response_mime_type="application/json" if json_mode else "text/plain",
                )
                response = get_gemini_client().models.generate_content(
                    model=model, contents=prompt, config=config
                )
                if response.text:
                    return response.text
            except Exception as error:
                last_error = error
                message = str(error)
                if "404" in message or "NOT_FOUND" in message:
                    break  # Model unavailable; move on to the next one
                wait_seconds = 20 * (attempt + 1)
                log(f"Gemini {model} problem (attempt {attempt + 1}): "
                    f"{message[:120]} ... waiting {wait_seconds}s")
                time.sleep(wait_seconds)
    raise Exception(f"Gemini failed on all models: {str(last_error)[:200]}")


def parse_json(text):
    """Parse a JSON response from Gemini, ignoring any surrounding code fences."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    return json.loads(text)


# =====================================================================
# STEP 1: NEWS COLLECTION (RSS)
# =====================================================================
def tag_name(tag):
    """Return the lowercase tag name without its XML namespace."""
    return tag.split("}")[-1].lower()


def parse_date(value):
    """Convert a date string to a timezone-aware datetime, or None if it is invalid."""
    if not value:
        return None
    value = value.strip()
    try:
        date = parsedate_to_datetime(value)
    except Exception:
        try:
            date = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return None
    if date.tzinfo is None:
        date = date.replace(tzinfo=dt.timezone.utc)
    return date


def parse_feed(xml_bytes, feed_url):
    """Parse an RSS or Atom feed and return a list of story dictionaries."""
    stories = []
    root = ET.fromstring(xml_bytes)
    feed_host = urlparse(feed_url).netloc.replace("www.", "")

    for element in root.iter():
        if tag_name(element.tag) not in ("item", "entry"):
            continue

        story = {"title": "", "link": "", "date": None, "summary": ""}
        for child in element:
            name = tag_name(child.tag)
            if name == "title":
                story["title"] = strip_html(child.text)
            elif name == "link":
                story["link"] = (child.text or child.attrib.get("href", "")).strip()
            elif name in ("pubdate", "date", "published", "updated") and not story["date"]:
                story["date"] = parse_date(child.text)
            elif name in ("description", "summary", "content") and not story["summary"]:
                story["summary"] = strip_html(child.text)[:600]

        # Ignore incomplete entries
        if not (story["title"] and story["link"] and story["date"]):
            continue

        # Google News appends the publisher to each title ("Title - Publisher")
        if "news.google.com" in feed_host and " - " in story["title"]:
            story["title"], story["source"] = story["title"].rsplit(" - ", 1)
        else:
            story["source"] = feed_host
        stories.append(story)

    return stories


def get_candidates(feeds, used_links, used_titles):
    """Collect recent, unused stories from all feeds, newest first."""
    now = dt.datetime.now(dt.timezone.utc)
    candidates, seen_titles = [], set()

    for url in feeds:
        try:
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=25)
            response.raise_for_status()
            found = parse_feed(response.content, url)
            log(f"Feed OK: {url[:70]} -> {len(found)} items")
        except Exception as error:
            log(f"Feed failed: {url[:70]} -> {str(error)[:80]}")
            continue

        for story in found:
            age_days = (now - story["date"]).total_seconds() / 86400
            title_key = story["title"].lower()
            is_out_of_range = age_days < 0 or age_days > MAX_AGE_DAYS
            is_already_used = story["link"] in used_links or title_key in used_titles
            if is_out_of_range or is_already_used or title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            story["age_days"] = round(age_days, 1)
            candidates.append(story)

    candidates.sort(key=lambda s: s["date"], reverse=True)
    return candidates[:MAX_CANDIDATES]


def read_article(story):
    """Return the article text, falling back to the RSS summary if it cannot be read."""
    link = story["link"]
    text = ""

    # Google News links are redirects, so only direct publisher links are fetched.
    if "news.google.com" not in link:
        try:
            response = requests.get(link, headers=REQUEST_HEADERS, timeout=25)
            paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", response.text, flags=re.S | re.I)
            paragraphs = [strip_html(p) for p in paragraphs]
            text = " ".join(p for p in paragraphs if len(p) > 60)[:5000]
        except Exception as error:
            log("Could not read article:", str(error)[:80])

    if len(text) < 300:
        text = f"{story['title']}. {story['summary']}"
    return text


def pick_story(topic, candidates):
    """Ask Gemini to select the most suitable story from the candidates."""
    menu = "\n".join(
        f"{i}. [{c['age_days']}d old] {c['title']} ({c['source']})"
        for i, c in enumerate(candidates)
    )
    prompt = (
        f"Topic today: {topic}.\n"
        "Pick the ONE story that is most interesting and relevant "
        "for a biotech student who follows healthcare AI. "
        "Prefer serious sources, avoid rumours.\n"
        f"{menu}\n"
        'Return JSON only: {"pick": number}'
    )
    answer = parse_json(ask_gemini(prompt, json_mode=True))
    index = int(answer["pick"])
    if not 0 <= index < len(candidates):
        index = 0  # Fall back to the newest story if the index is invalid
    return candidates[index]


# =====================================================================
# STEP 2: POST WRITING AND FACT-CHECKING
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


def write_post(topic, story, article_text, today, problems=None):
    """
    Generate a LinkedIn post from a story.

    When `problems` is provided, the model is asked to correct those
    issues reported by the fact-checker.
    """
    fix_note = ""
    if problems:
        fix_note = (
            "\nA fact-checker found these problems in your previous draft. Fix them:\n- "
            + "\n- ".join(problems)
        )

    prompt = f"""{STYLE_RULES}
Today's date: {today.strftime('%d %B %Y')}. Story published: {story['date'].strftime('%d %B %Y')} ({story['age_days']} days ago).
Only use words like "today", "just", "this week" if the story is 2 days old or less; otherwise say the month or "recently".
Topic of the day: {topic}
STORY TITLE: {story['title']}
PUBLISHER: {story['source']}
SOURCE TEXT:
{article_text}
{fix_note}
Return JSON only:
{{"post": "the post text", "hashtags": ["#AI", "..."] (3 to 5), "headline": "max 9 words for an image card",
 "points": ["3 short takeaways, each max 12 words"], "publisher": "publisher name"}}"""
    return parse_json(ask_gemini(prompt, json_mode=True))


def write_weekly_recap(history, today):
    """
    Generate the Sunday recap from this week's published posts.

    Returns (post_data, source_text), or (None, None) when fewer than
    two posts are available.
    """
    week_start = (today - dt.timedelta(days=7)).strftime("%Y-%m-%d")
    week_posts = [
        h for h in history
        if h.get("status") == "posted" and h.get("date", "") >= week_start
    ]
    if len(week_posts) < 2:
        return None, None

    source_text = "\n\n".join(
        f"Post on {h['date']} ({h.get('topic', '')}): {h.get('post', '')}"
        for h in week_posts
    )
    prompt = f"""{STYLE_RULES}
This is a WEEKLY RECAP post. Use ONLY the posts below (from this week). Add no new facts.
Mention what the week's posts covered and one common theme. End with a question.
POSTS FROM THIS WEEK:
{source_text}
Return JSON only:
{{"post": "the recap text", "hashtags": ["#AI", "..."] (3 to 5), "headline": "max 9 words",
 "points": ["3 short takeaways, each max 12 words"], "publisher": "my posts this week"}}"""
    return parse_json(ask_gemini(prompt, json_mode=True)), source_text


def fact_check(post, source_text):
    """Verify the post against its source; return {"ok": bool, "problems": [...]}."""
    prompt = f"""You are a strict fact-checker.
SOURCE TEXT:
{source_text}

POST:
{post}

Check every factual claim about the story (numbers, names, dates, who did what) against the SOURCE TEXT.
Opinions and questions are fine. Statements about the author are fine only if they match: {ABOUT_ME}
Return JSON only: {{"ok": true or false, "problems": ["each unsupported claim, short"]}}"""
    return parse_json(ask_gemini(prompt, json_mode=True))


# =====================================================================
# STEP 3: IMAGE GENERATION
# =====================================================================
# Accent colors for the summary card, one per weekday.
CARD_COLORS = ["#0F766E", "#3B4CCA", "#B4433A", "#7A4FB5", "#1D6FA5", "#2F7D32", "#C77700"]

# Font locations on the GitHub Actions (Ubuntu) runner, in order of preference.
BOLD_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]
REGULAR_FONT_PATHS = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def load_font(size, bold=False):
    """Load the first available system font, or Pillow's default font as a fallback."""
    for path in (BOLD_FONT_PATHS if bold else REGULAR_FONT_PATHS):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def wrap_text(draw, text, font, max_width, max_lines):
    """Split text into lines that fit max_width, truncating with '...' beyond max_lines."""
    lines, current = [], ""
    for word in text.split():
        attempt = (current + " " + word).strip()
        if draw.textlength(attempt, font=font) <= max_width:
            current = attempt
        else:
            lines.append(current)
            current = word
    lines.append(current)

    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(" .,") + "..."
    return lines


def make_summary_card(headline, points, publisher, weekday):
    """
    Render a 1200x627 summary card (headline, three key points, source).

    The card is drawn locally, so it has no cost and no API quota.
    Returns PNG bytes, or None if Pillow is not installed.
    """
    if Image is None:
        return None

    scale = 2  # Render at double size, then downscale for sharper text
    width, height = 1200 * scale, 627 * scale
    color = CARD_COLORS[weekday % len(CARD_COLORS)]

    image = Image.new("RGB", (width, height), "#F6F7F9")
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 14 * scale, height], fill=color)  # Accent bar on the left

    # Headline
    y = 52 * scale
    title_font = load_font(50 * scale, bold=True)
    for line in wrap_text(draw, headline, title_font, 1060 * scale, 3):
        draw.text((64 * scale, y), line, font=title_font, fill="#1B2430")
        y += 62 * scale
    y += 26 * scale

    # Key points
    point_font = load_font(28 * scale)
    for point in (points or [])[:3]:
        draw.ellipse([64 * scale, y + 11 * scale, 78 * scale, y + 25 * scale], fill=color)
        for line in wrap_text(draw, point, point_font, 1010 * scale, 2):
            draw.text((96 * scale, y), line, font=point_font, fill="#1B2430")
            y += 38 * scale
        y += 20 * scale

    # Footer: source on the left, disclosure label on the right
    footer_font = load_font(18 * scale)
    footer_y = 583 * scale
    draw.text((64 * scale, footer_y), f"Source: {publisher}"[:70],
              font=footer_font, fill="#5B6675")
    label = "Summary card made with AI"
    label_x = width - 64 * scale - draw.textlength(label, font=footer_font)
    draw.text((label_x, footer_y), label, font=footer_font, fill="#5B6675")

    image = image.resize((1200, 627), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_ai_image(headline):
    """Generate an image with a Gemini image model; return None if all models fail."""
    from google.genai import types

    prompt = (
        "Clean, modern, professional editorial illustration for a LinkedIn post about: "
        f"{headline}. No text, no letters, no logos, no real people."
    )
    for model in IMAGE_MODELS:
        try:
            response = get_gemini_client().models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
            )
            for part in response.candidates[0].content.parts:
                if getattr(part, "inline_data", None) and part.inline_data.data:
                    return part.inline_data.data
        except Exception as error:
            log(f"AI image {model} failed: {str(error)[:120]}")
    return None


# =====================================================================
# STEP 4: LINKEDIN PUBLISHING (official API)
# =====================================================================
def linkedin_headers():
    """
    Build the LinkedIn API request headers.

    LinkedIn only accepts recent API versions, so the version is set to
    the month two months ago, which is always supported.
    """
    version = (dt.datetime.now() - dt.timedelta(days=60)).strftime("%Y%m")
    return {
        "Authorization": f"Bearer {os.environ['LI_TOKEN']}",
        "LinkedIn-Version": version,
        "X-Restli-Protocol-Version": "2.0.0",
        "Content-Type": "application/json",
    }


def linkedin_escape(text):
    """Escape characters that LinkedIn treats as markup. '#' is kept so hashtags still work."""
    for char in "\\|{}@[]()<>*_~":
        text = text.replace(char, "\\" + char)
    return text


def upload_linkedin_image(headers, author_urn, image_bytes):
    """Upload an image to LinkedIn and return its image identifier."""
    init = requests.post(
        "https://api.linkedin.com/rest/images?action=initializeUpload",
        headers=headers,
        json={"initializeUploadRequest": {"owner": author_urn}},
        timeout=30,
    )
    init.raise_for_status()
    value = init.json()["value"]

    upload = requests.put(
        value["uploadUrl"],
        data=image_bytes,
        headers={"Authorization": headers["Authorization"]},
        timeout=60,
    )
    upload.raise_for_status()
    return value["image"]


def post_to_linkedin(text, image_bytes, alt_text):
    """Publish the post to the personal LinkedIn profile and return its URL."""
    headers = linkedin_headers()

    # Identify the profile that owns the access token
    me = requests.get(
        "https://api.linkedin.com/v2/userinfo",
        headers={"Authorization": headers["Authorization"]},
        timeout=30,
    )
    if me.status_code == 401:
        raise Exception(
            "LinkedIn token expired or invalid. Generate a new token and update the LI_TOKEN secret."
        )
    me.raise_for_status()
    author_urn = f"urn:li:person:{me.json()['sub']}"

    body = {
        "author": author_urn,
        "commentary": linkedin_escape(text),
        "visibility": "PUBLIC",
        "distribution": {
            "feedDistribution": "MAIN_FEED",
            "targetEntities": [],
            "thirdPartyDistributionChannels": [],
        },
        "lifecycleState": "PUBLISHED",
        "isReshareDisabledByAuthor": False,
    }

    if image_bytes:
        image_id = upload_linkedin_image(headers, author_urn, image_bytes)
        body["content"] = {
            "media": {
                "title": alt_text[:100] or "Image",
                "altText": alt_text[:300],
                "id": image_id,
            }
        }

    response = requests.post(
        "https://api.linkedin.com/rest/posts", headers=headers, json=body, timeout=60
    )
    if response.status_code not in (200, 201):
        raise Exception(f"LinkedIn post failed {response.status_code}: {response.text[:300]}")

    post_id = response.headers.get("x-restli-id", "")
    return f"https://www.linkedin.com/feed/update/{post_id}" if post_id else ""


# =====================================================================
# MAIN WORKFLOW
# =====================================================================
def create_draft(history, today, entry):
    """
    Select a story and generate the first draft.

    Returns (topic, post_data, source_text, story). The story is None
    for weekly recaps, which are not based on a single article.
    """
    weekday = today.weekday()
    topic, feeds = TOPICS[weekday]
    entry["topic"] = topic

    # Stories already published must not be reused
    posted = [h for h in history if h.get("status") == "posted"]
    used_links = {h.get("source_url") for h in posted}
    used_titles = {h.get("title", "").lower() for h in posted}

    # Sunday: attempt a weekly recap first
    if weekday == 6:
        data, source_text = write_weekly_recap(history, today)
        if data is not None:
            return topic, data, source_text, None
        # Too few posts for a recap; use a regular biology story instead
        topic, feeds = TOPICS[BIOLOGY_TOPIC_DAY]
        entry["topic"] = topic

    candidates = get_candidates(feeds, used_links, used_titles)
    if not candidates:
        raise Skip("No fresh (last 7 days) unused stories found in the feeds.")

    story = pick_story(topic, candidates)
    source_text = read_article(story)
    entry["title"] = story["title"]
    entry["source_url"] = story["link"]
    log("Chosen story:", story["title"], "|", story["source"], f"| {story['age_days']} days old")

    data = write_post(topic, story, source_text, today)
    return topic, data, source_text, story


def verify_draft(topic, data, source_text, story, today):
    """
    Fact-check the draft, allowing one rewrite.

    Raises Skip if the post still fails verification, or if it is a
    recap (which has no single source article to rewrite from).
    """
    check = fact_check(data["post"], source_text)
    if check.get("ok"):
        return data

    problems = check.get("problems", [])
    log("Fact-check problems:", problems)

    if story is not None:
        data = write_post(topic, story, source_text, today, problems=problems)
        check = fact_check(data["post"], source_text)
        if check.get("ok"):
            return data

    reasons = "; ".join(check.get("problems", []))[:400]
    raise Skip("Fact-check failed: " + reasons)


def build_final_text(data, story):
    """Combine the post, source line, and hashtags. Returns (text, publisher)."""
    hashtags = " ".join(
        tag if tag.startswith("#") else "#" + tag
        for tag in data.get("hashtags", [])[:5]
    )
    publisher = data.get("publisher") or (story["source"] if story else "")

    source_line = ""
    if story is not None:
        source_line = f"Source: {publisher}"
        # Google News links are redirects, so they are not shown to readers
        if "news.google.com" not in story["link"]:
            source_line += f" - {story['link']}"

    parts = [data["post"].strip(), source_line, hashtags]
    return "\n\n".join(part for part in parts if part), publisher


def prepare_image(data, topic, publisher, weekday, today, entry):
    """Create the post image and save a copy to the images folder."""
    headline = data.get("headline", topic)

    image = make_ai_image(headline) if USE_AI_IMAGE else None
    if image is None:
        image = make_summary_card(headline, data.get("points", []), publisher, weekday)

    if image is None:
        telegram_text("Pillow is not installed; posting without an image. "
                      "Add 'pillow' to requirements.txt.")
        return None

    os.makedirs(IMAGE_DIR, exist_ok=True)
    path = f"{IMAGE_DIR}/{today.strftime('%Y-%m-%d')}.png"
    with open(path, "wb") as file:
        file.write(image)
    entry["image"] = path
    return image


def run(history, today, entry):
    """Execute the daily job: draft, verify, create image, then preview or publish."""
    topic, data, source_text, story = create_draft(history, today, entry)
    data = verify_draft(topic, data, source_text, story, today)

    final_text, publisher = build_final_text(data, story)
    entry["post"] = final_text

    image = prepare_image(data, topic, publisher, today.weekday(), today, entry)

    if PREVIEW_ONLY:
        entry["status"] = "preview"
        telegram_text("PREVIEW (nothing posted to LinkedIn)\nTopic: " + topic + "\n\n" + final_text)
        telegram_photo(image, "Image for the preview post")
        log("PREVIEW sent to Telegram.")
        return

    link = post_to_linkedin(final_text, image, data.get("headline", topic))
    entry["status"] = "posted"
    entry["linkedin_url"] = link
    telegram_text(f"Posted on LinkedIn today ({topic}).\n{link}")
    log("Posted:", link)


def main():
    """Run the daily job, record the outcome in history.json, and set the exit code."""
    today = dt.datetime.now(IST)
    history = load_history()
    entry = {
        "date": today.strftime("%Y-%m-%d"),
        "time": today.strftime("%H:%M"),
        "topic": "",
        "status": "",
        "title": "",
        "source_url": "",
        "post": "",
        "image": "",
        "linkedin_url": "",
        "reason": "",
    }

    exit_code = 0
    try:
        run(history, today, entry)
    except Skip as reason:
        entry["status"], entry["reason"] = "skipped", str(reason)
        telegram_text(f"Post SKIPPED today.\nReason: {reason}")
        log("SKIPPED:", reason)
        exit_code = 1
    except Exception as error:
        entry["status"], entry["reason"] = "failed", str(error)[:400]
        telegram_text(f"Robot FAILED today.\nReason: {str(error)[:500]}")
        log("FAILED:", error)
        exit_code = 1

    history.append(entry)
    save_history(history)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
