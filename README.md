# Daily Assistant

An automated agent that researches a topic, writes a LinkedIn post with Gemini, fact-checks it, and publishes it every morning. It runs for free on GitHub Actions.

**Live dashboard:** https://kaushikaiml.github.io/daily-assistant

## Features

- **Fresh research:** reads RSS feeds (last 7 days only) and never reuses a story that was already posted.
- **AI writing:** Gemini picks the story, writes the post, and keeps to facts found in the source text.
- **Fact-check gate:** a second Gemini pass verifies every claim. If a claim is unsupported, the post is skipped and an alert is sent to Telegram.
- **Summary image:** a clean card is drawn with Pillow (headline and three key points), so no paid image API is needed.
- **Preview mode:** send the draft to Telegram first, and switch to live posting when you are happy with it.
- **History dashboard:** a static GitHub Pages site shows every run with its status, source, and post text.

## How it works

1. GitHub Actions starts the agent on a daily schedule.
2. The agent collects stories for the day's topic and picks one.
3. It reads the article and writes the post.
4. The fact-checker reviews the post (one rewrite is allowed).
5. The summary image is created.
6. The post is published through the official LinkedIn API, or sent to Telegram in preview mode.
7. The result is saved to `history.json`, which feeds the dashboard.

## Weekly topic plan

| Day | Topic |
|-----|-------|
| Monday | Current AI |
| Tuesday | AI in healthcare |
| Wednesday | Biotechnology |
| Thursday | Technology |
| Friday | Recent discoveries in biology |
| Saturday | AI meets biotech |
| Sunday | Weekly recap, built only from that week's own posts |

## Tech stack

Python 3.11, Google Gemini API (`google-genai`), Pillow, Requests, LinkedIn API, Telegram Bot API, GitHub Actions, GitHub Pages.

## Project structure

```
daily-assistant/
├── .github/workflows/daily.yml   # Schedule and run steps
├── agent.py                      # The agent
├── index.html                    # History dashboard
├── history.json                  # Run history (updated automatically)
├── requirements.txt              # Python dependencies
├── LICENSE
└── README.md
```

## Setup

1. **Fork or clone** this repository.
2. **Create a Gemini API key** in Google AI Studio.
3. **Create a LinkedIn developer app** with the products needed to post on behalf of your profile, then generate an access token.
4. **Create a Telegram bot** with BotFather and note your chat ID.
5. **Add the secrets** in the repository under *Settings > Secrets and variables > Actions*:

   | Secret | Purpose |
   |--------|---------|
   | `GEMINI_KEY` | Gemini API key |
   | `LI_TOKEN` | LinkedIn access token |
   | `TG_TOKEN` | Telegram bot token |
   | `TG_CHAT_ID` | Telegram chat that receives previews and alerts |

6. **Enable GitHub Pages** under *Settings > Pages* (deploy from the `main` branch) to publish the dashboard.
7. Edit `ABOUT_ME` in `agent.py` so it contains only true facts about you.
8. Keep `PREVIEW_ONLY = True` for the first runs, check the drafts in Telegram, then set it to `False`.

Never commit API keys or tokens to the repository. Keep them in GitHub Secrets only.

## Configuration

Main settings are at the top of `agent.py`:

| Setting | Meaning |
|---------|---------|
| `PREVIEW_ONLY` | `True` sends drafts to Telegram only. `False` posts live. |
| `USE_AI_IMAGE` | `False` draws the card with code (free). `True` tries Gemini image models. |
| `TEXT_MODELS` | Gemini models in order of preference, with automatic fallback. |
| `MAX_AGE_DAYS` | Maximum age of a story in days. |
| `ABOUT_ME` | The only personal facts the AI may use. |

The schedule is set in `.github/workflows/daily.yml` (cron uses UTC; `30 1 * * *` is 07:00 IST). You can also run it manually from the *Actions* tab.

## Notes and limitations

- **Token expiry:** LinkedIn access tokens last about 60 days. When the token expires, the run fails with a clear message and a Telegram alert; generate a new token and update `LI_TOKEN`.
- **Schedule delays:** GitHub may start scheduled workflows later than the exact time during busy periods.
- **Disclosure:** each generated image carries the label "Summary card made with AI".
- **Responsibility:** the fact-checker reduces errors but cannot guarantee accuracy. Review previews regularly.

## License

Released under the MIT License. See `LICENSE`.
