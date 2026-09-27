# AS Chatbot — Netlify-native version

Everything runs on Netlify: a static chat page + one serverless Function
that calls a hosted model and self-updates memory in **Netlify Blobs**
(Netlify's built-in persistent key/value store). No separate backend to
host or manage.

## Why it works this way

Your actual fine-tuned model (Qwen2.5-1.5B + LoRA, ~3GB merged) can't run
inside a Netlify Function — Functions are AWS Lambda under the hood: no
GPU, a 50MB code limit, and a hard timeout. So the Function calls your
model hosted on Hugging Face instead, and does everything else itself:

```
public/index.html  --POST /api/chat-->  netlify/functions/chat.js
                                                |        |
                                    Netlify Blobs      Hugging Face
                                 (memory, persists)   (generates replies)
```

`chat.js` on every message:
1. loads facts + conversation history from Netlify Blobs (seeded from your
   real backup on first run — `netlify/functions/seed-data.js`)
2. retrieves relevant facts by keyword overlap (a lighter stand-in for the
   original sentence-embedding search — no room to run an embedding model
   here either)
3. builds a ChatML prompt (Qwen's format) and calls your hosted model
4. asks the model a second, silent question to pull out any new fact from
   what was just said
5. **writes new facts straight back into Blobs** — this is the
   self-updating part, and it happens entirely on Netlify's infrastructure

Tested against your real data before shipping: retrieval correctly surfaces
the lilies fact for a lilies-related question, dedup correctly recognizes
paraphrases of facts you already have ("loves lilies very much" vs "You
love lilies.") while letting genuinely new facts through, and the ChatML
formatting matches `as-chatbot-final/chat_template.jinja` exactly.

## One-time setup (do this before deploying)

**1. Get your model onto Hugging Face** — the Function needs an HTTP
endpoint to call; it can't load `as-chatbot-final/` directly.
```bash
pip install transformers peft torch huggingface_hub
huggingface-cli login
python merge_and_push.py --repo-id your-username/as-chatbot-merged --private
```
This merges your LoRA adapter into the base model and pushes it to your
account. Use `--private` — this model was trained on real personal
conversations.

**2. Deploy that model somewhere callable.** Two options, see
`merge_and_push.py`'s docstring for detail:
- **Inference Endpoints** (recommended): reliable, works with a custom
  private model, can scale to zero when idle. Gives you a URL like
  `https://xxxxx.aws.endpoints.huggingface.cloud`.
- **Free Serverless API**: `https://api-inference.huggingface.co/models/your-username/as-chatbot-merged` — free, but not guaranteed to keep a custom model warm; fine for testing, not for something you want reliably live.

**3. Get an HF API token** with inference access, from your Hugging Face
account settings.

## Deploying to Netlify

1. Push this whole folder to a GitHub repo (or drag-and-drop deploy — Netlify's UI supports both).
2. In Netlify: New site → connect the repo. It reads `netlify.toml` automatically — publish dir `public`, functions dir `netlify/functions`.
3. **Site settings → Environment variables**, set:
   - `HF_API_URL` — the endpoint URL from step 2 above
   - `HF_API_TOKEN` — your Hugging Face token
   - `CHAT_API_KEY` — a random string of your choosing, to lock the chat endpoint down (recommended — set it, and also set `window.AS_CHATBOT_API_KEY` to the same value near the top of `public/index.html`'s `<script>` block before deploying)
4. Enable **Blobs** for the site — it's on by default for new sites on current Netlify plans; if `getStore` errors on first call, check Site settings → Blobs is enabled.
5. Deploy. First message will seed Blobs from your real backup data automatically.

## Trying it locally first

```bash
npm install -g netlify-cli
netlify dev
```
This runs the Function + static site together locally, using Netlify's
local Blobs emulation, so you can test before actually deploying.

## Important: this bot's memory holds real personal data

The seeded memory (and everything it learns afterward) includes full
names, a date of birth, a home city, and family details about a real,
identifiable person. Once deployed:
- Set `CHAT_API_KEY` — don't leave the endpoint open to anyone with the URL.
- Keep the Hugging Face repo **private** (`--private` in step 1).
- The self-update loop means Blobs keeps growing with real personal
  details over time. You can inspect what it's learned anytime from
  Netlify's dashboard → your site → Blobs → `as-chatbot-memory` store →
  the `facts` key.

## Files

| Path | Purpose |
|---|---|
| `public/index.html` | The entire frontend — one file |
| `netlify/functions/chat.js` | The entire backend — one function: retrieval, prompting, model call, self-update |
| `netlify/functions/seed-data.js` | Your real permanent + semantic memory, used only to initialize Blobs on first run |
| `netlify.toml` | Wires up publish dir, functions dir, and the `/api/chat` route |
| `package.json` | Declares `@netlify/blobs` so Netlify bundles it with the Function |
| `merge_and_push.py` | One-time script (run locally) to get your model onto Hugging Face |

## Trade-off vs. running your exact local weights

This version trades "runs your exact local weights on a server you
control" for "everything lives in one Netlify deploy, no separate host to
manage." Retrieval here is keyword-based instead of embedding-based (no
room to run an embedding model in a Function either), and replies depend
on Hugging Face being up. If you'd rather self-host the model on a proper
backend instead of calling Hugging Face, that's a different, separately
deployed setup — just ask if you want that version instead.
