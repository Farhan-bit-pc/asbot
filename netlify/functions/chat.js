// netlify/functions/chat.js
//
// The whole backend, in one function:
//   1. loads memory from Netlify Blobs (seeding it from seed-data.js on first run)
//   2. retrieves relevant facts for the user's message (lexical scoring — see note below)
//   3. builds a ChatML prompt (Qwen's format) and calls your hosted model API
//   4. asks the model a second, silent question to extract any new facts
//   5. writes new facts + updated conversation history back to Blobs
//
// Netlify Blobs is what makes this "self-updating while staying on Netlify":
// it's a real persistent, edge-replicated key/value store built into Netlify,
// available from any function with zero extra setup — no separate database needed.
//
// IMPORTANT — what this can't do that the local-model version could:
// Your actual fine-tuned LoRA weights can't run here (no GPU, no room for a
// 3GB model in a 50MB function). This calls a model hosted on Hugging Face
// instead — see README.md "One-time setup" for how to get your fine-tune
// there. Retrieval here is also simpler: keyword overlap, not the original
// sentence-embedding cosine similarity (no room to run an embedding model
// either). It's a reasonable approximation for a personal-facts store this
// size, not identical to the original.

const { getStore } = require("@netlify/blobs");
const { SEED_DATA } = require("./seed-data");

const HF_API_URL = process.env.HF_API_URL; // e.g. https://api-inference.huggingface.co/models/your-username/as-chatbot-merged
const HF_API_TOKEN = process.env.HF_API_TOKEN;
const SHARED_API_KEY = process.env.CHAT_API_KEY; // optional shared secret the frontend must send
const SESSION_GAP_MINUTES = 30;
const MAX_HISTORY_MESSAGES = 8;
const TOP_K_FACTS = 5;

// ---------------------------------------------------------------------------
// memory helpers (Netlify Blobs)
// ---------------------------------------------------------------------------

async function loadMemory(store) {
  let permanent = await store.get("permanent", { type: "json" });
  let facts = await store.get("facts", { type: "json" });
  if (!permanent || !facts) {
    permanent = SEED_DATA.permanent;
    facts = SEED_DATA.facts;
    await store.setJSON("permanent", permanent);
    await store.setJSON("facts", facts);
  }
  return { permanent, facts };
}

async function loadHistory(store, sessionId) {
  const key = `history:${sessionId}`;
  const record = await store.get(key, { type: "json" });
  const now = Date.now();
  if (!record) return { key, messages: [], lastActivity: now };

  const gapMinutes = (now - record.lastActivity) / 60000;
  if (gapMinutes > SESSION_GAP_MINUTES) {
    return { key, messages: [], lastActivity: now }; // fresh session
  }
  return { key, messages: record.messages, lastActivity: now };
}

async function saveHistory(store, key, messages, lastActivity) {
  const trimmed = messages.slice(-MAX_HISTORY_MESSAGES);
  await store.setJSON(key, { messages: trimmed, lastActivity });
  return trimmed;
}

function permanentAsPromptBlock(permanent) {
  return Object.values(permanent).flat().map((f) => `- ${f}`).join("\n");
}

// crude but dependency-free lexical retrieval: score by shared-word overlap
function retrieveFacts(query, facts, topK = TOP_K_FACTS) {
  const queryWords = new Set(query.toLowerCase().match(/[a-z0-9']+/g) || []);
  if (queryWords.size === 0) return [];

  const scored = facts.map((f) => {
    const factWords = new Set(f.fact.toLowerCase().match(/[a-z0-9']+/g) || []);
    let overlap = 0;
    for (const w of queryWords) if (factWords.has(w)) overlap++;
    return { fact: f, score: overlap };
  });

  return scored
    .filter((s) => s.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map((s) => s.fact);
}

function isDuplicateFact(newFactText, existingFacts) {
  const stopwords = new Set(["you", "your", "the", "a", "an", "is", "are", "was", "were", "to", "and", "very", "much", "of", "in", "on"]);
  const words = (s) => new Set((s.toLowerCase().match(/[a-z0-9']+/g) || []).filter((w) => !stopwords.has(w)));
  const target = words(newFactText);
  if (target.size === 0) return false;

  return existingFacts.some((f) => {
    const existing = words(f.fact);
    if (existing.size === 0) return false;
    let overlap = 0;
    for (const w of target) if (existing.has(w)) overlap++;
    const similarity = overlap / Math.min(target.size, existing.size);
    return similarity >= 0.5; // at least half the smaller set's meaningful words match
  });
}

// ---------------------------------------------------------------------------
// prompting (replicates as-chatbot-final/chat_template.jinja — Qwen ChatML)
// ---------------------------------------------------------------------------

function toChatML(messages) {
  let out = "";
  for (const m of messages) {
    out += `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`;
  }
  out += "<|im_start|>assistant\n";
  return out;
}

function buildSystemPrompt(personaName, permanent, retrievedFacts) {
  const retrievedBlock = retrievedFacts.length
    ? retrievedFacts.map((f) => `- ${f.fact}`).join("\n")
    : "(nothing specifically relevant retrieved)";
  return (
    `You are ${personaName}. Stay fully in character, respond the way ` +
    `${personaName} would based on the facts below.\n\n` +
    `## Core facts (always true)\n${permanentAsPromptBlock(permanent)}\n\n` +
    `## Relevant to this conversation\n${retrievedBlock}\n`
  );
}

const FACT_EXTRACTION_SYSTEM_PROMPT = `You are a silent fact-logging assistant. You do not chat.
Given one message, decide whether it states a new, durable, personal fact about the speaker
(identity, preferences, family, plans, events, opinions, likes/dislikes). Ignore small talk,
questions, jokes, and anything that isn't a durable fact.
Respond with ONLY a JSON array (no prose, no markdown fences). Each item:
{"fact": "<fact stated in third person>", "topic": "<identity|education|family|location|preferences|relationship|plans|other>"}
If there is no durable fact, respond with exactly: []`;

// ---------------------------------------------------------------------------
// hosted model calls
// ---------------------------------------------------------------------------

async function callModel(promptText, maxNewTokens) {
  const res = await fetch(HF_API_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${HF_API_TOKEN}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      inputs: promptText,
      parameters: {
        max_new_tokens: maxNewTokens,
        temperature: 0.8,
        top_p: 0.9,
        return_full_text: false,
      },
    }),
  });

  if (!res.ok) {
    const errText = await res.text().catch(() => "");
    throw new Error(`Model API returned ${res.status}: ${errText.slice(0, 300)}`);
  }

  const data = await res.json();
  // Serverless Inference API returns [{generated_text: "..."}]; some Inference
  // Endpoints return a bare object instead — handle both shapes.
  if (Array.isArray(data)) return (data[0]?.generated_text || "").trim();
  if (data.generated_text) return data.generated_text.trim();
  throw new Error(`Unrecognized model API response shape: ${JSON.stringify(data).slice(0, 300)}`);
}

async function extractFacts(speaker, message) {
  if (message.trim().length < 8) return [];
  const chat = [
    { role: "system", content: FACT_EXTRACTION_SYSTEM_PROMPT },
    { role: "user", content: `Speaker: ${speaker}\nMessage: "${message}"` },
  ];
  let raw;
  try {
    raw = await callModel(toChatML(chat), 150);
  } catch {
    return []; // extraction failing should never break the chat reply
  }
  const match = raw.match(/\[[\s\S]*\]/);
  if (!match) return [];
  try {
    const items = JSON.parse(match[0]);
    return items
      .filter((it) => it && typeof it.fact === "string" && it.fact.trim())
      .map((it) => ({ fact: it.fact.trim(), topic: it.topic || "other" }));
  } catch {
    return [];
  }
}

// ---------------------------------------------------------------------------
// handler
// ---------------------------------------------------------------------------

exports.handler = async (event) => {
  if (event.httpMethod !== "POST") {
    return { statusCode: 405, body: "Method not allowed" };
  }
  if (SHARED_API_KEY) {
    const provided = event.headers["x-api-key"];
    if (provided !== SHARED_API_KEY) {
      return { statusCode: 401, body: JSON.stringify({ error: "Invalid or missing API key" }) };
    }
  }
  if (!HF_API_URL || !HF_API_TOKEN) {
    return {
      statusCode: 500,
      body: JSON.stringify({ error: "HF_API_URL / HF_API_TOKEN not configured — see README.md" }),
    };
  }

  let payload;
  try {
    payload = JSON.parse(event.body || "{}");
  } catch {
    return { statusCode: 400, body: JSON.stringify({ error: "Invalid JSON body" }) };
  }

  const speaker = (payload.speaker || "").trim();
  const message = (payload.message || "").trim();
  const sessionId = payload.sessionId || "default";

  if (!["Farhan", "Ameera"].includes(speaker)) {
    return { statusCode: 400, body: JSON.stringify({ error: "speaker must be 'Farhan' or 'Ameera'" }) };
  }
  if (!message) {
    return { statusCode: 400, body: JSON.stringify({ error: "message is empty" }) };
  }

  const personaName = speaker === "Farhan" ? "Ameera" : "Farhan"; // bot plays the other partner

  try {
    const store = getStore("as-chatbot-memory", {
      siteID: process.env.NETLIFY_SITE_ID,
      token: process.env.NETLIFY_BLOBS_TOKEN,
    });
    const { permanent, facts } = await loadMemory(store);
    const { key: historyKey, messages: history, lastActivity } = await loadHistory(store, sessionId);

    const retrieved = retrieveFacts(message, facts);
    const systemPrompt = buildSystemPrompt(personaName, permanent, retrieved);

    const chat = [{ role: "system", content: systemPrompt }, ...history, { role: "user", content: message }];
    const replyText = await callModel(toChatML(chat), 200);

    const updatedHistory = [...history, { role: "user", content: message }, { role: "assistant", content: replyText }];
    await saveHistory(store, historyKey, updatedHistory, Date.now());

    // self-update: extract + store any new facts, best-effort
    let learned = [];
    try {
      const extracted = await extractFacts(speaker, message);
      const newFacts = extracted.filter((f) => !isDuplicateFact(f.fact, facts));
      if (newFacts.length) {
        const category = `about_${speaker.toLowerCase()}`;
        const updatedFacts = [
          ...facts,
          ...newFacts.map((f) => ({ subject: speaker, category, fact: f.fact, topic: f.topic })),
        ];
        const updatedPermanent = { ...permanent, [category]: [...(permanent[category] || []), ...newFacts.map((f) => f.fact)] };
        await store.setJSON("facts", updatedFacts);
        await store.setJSON("permanent", updatedPermanent);
        learned = newFacts.map((f) => f.fact);
      }
    } catch {
      // never let a failed extraction affect the reply already generated
    }

    return {
      statusCode: 200,
      body: JSON.stringify({ reply: replyText, learned }),
    };
  } catch (err) {
    return { statusCode: 502, body: JSON.stringify({ error: String(err.message || err) }) };
  }
};
