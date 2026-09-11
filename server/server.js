// Cosmos Smart Finder - backend
require("dotenv").config();
const express = require("express");
const cors = require("cors");
const rateLimit = require("express-rate-limit");
const fs = require("fs");
const path = require("path");
const EkpaSearch = require("../public/search-engine.js");

// Load Cosmosport data
const PRODUCTS = JSON.parse(fs.readFileSync(path.join(__dirname, "./data/products.json"), "utf-8"));
let CONCEPTS = JSON.parse(fs.readFileSync(path.join(__dirname, "./data/concepts.json"), "utf-8"));
const CONCEPTS_PATH = path.join(__dirname, "./data/concepts.json");

// Map products by URL
const PRODUCTS_BY_URL = {};
PRODUCTS.forEach((p) => { PRODUCTS_BY_URL[p.url] = p; });

// Analytics
const ANALYTICS_PATH = path.join(__dirname, "analytics.json");
const ANALYTICS_MAX_QUERIES = 300;
const ANALYTICS_FLUSH_MS = 30 * 1000;
let analyticsQueries = {};
try {
  analyticsQueries = JSON.parse(fs.readFileSync(ANALYTICS_PATH, "utf-8"));
} catch (e) {
  analyticsQueries = {};
}
let analyticsDirty = false;

function normalizeAnalyticsQuery(q) {
  return String(q || "").trim().toLowerCase().slice(0, 200);
}

function getOrCreateQueryEntry(q) {
  if (analyticsQueries[q]) return analyticsQueries[q];
  const keys = Object.keys(analyticsQueries);
  if (keys.length >= ANALYTICS_MAX_QUERIES) {
    let worstKey = keys[0];
    keys.forEach((k) => { if (analyticsQueries[k].count < analyticsQueries[worstKey].count) worstKey = k; });
    delete analyticsQueries[worstKey];
  }
  analyticsQueries[q] = { count: 0, zero_result: 0, clicks: {}, last_seen: null };
  return analyticsQueries[q];
}

function trackSearch(rawQuery, resultCount) {
  const q = normalizeAnalyticsQuery(rawQuery);
  if (!q) return;
  const entry = getOrCreateQueryEntry(q);
  entry.count += 1;
  if (resultCount === 0) entry.zero_result += 1;
  entry.last_seen = new Date().toISOString();
  analyticsDirty = true;
}

function trackClick(rawQuery, productUrl) {
  const q = normalizeAnalyticsQuery(rawQuery);
  const entry = getOrCreateQueryEntry(q || "(χωρίς query)");
  entry.clicks[productUrl] = (entry.clicks[productUrl] || 0) + 1;
  entry.last_seen = new Date().toISOString();
  analyticsDirty = true;
}

function flushAnalytics() {
  if (!analyticsDirty) return;
  analyticsDirty = false;
  try {
    fs.writeFileSync(ANALYTICS_PATH, JSON.stringify(analyticsQueries));
  } catch (err) {
    console.error("Αποτυχία αποθήκευσης analytics.json:", err);
  }
}
setInterval(flushAnalytics, ANALYTICS_FLUSH_MS);
for (const sig of ["SIGTERM", "SIGINT"]) {
  process.on(sig, () => { flushAnalytics(); process.exit(0); });
}

const PORT = process.env.PORT || 8787;
const PROVIDER = (process.env.LLM_PROVIDER || "gemini").toLowerCase();

// CORS
const DEFAULT_ORIGINS = ["http://localhost:8787", "http://localhost:3000"];
const envOrigins = (process.env.ALLOWED_ORIGINS || "").split(",").map((s) => s.trim()).filter(Boolean);
const ALLOWED_ORIGINS = envOrigins.length ? envOrigins : DEFAULT_ORIGINS;
const RAILWAY_PUBLIC_DOMAIN = process.env.RAILWAY_PUBLIC_DOMAIN ? `https://${process.env.RAILWAY_PUBLIC_DOMAIN}` : null;

function corsOriginCheck(origin, callback) {
  if (!origin) return callback(null, true);
  if (ALLOWED_ORIGINS.includes(origin) || origin === RAILWAY_PUBLIC_DOMAIN) return callback(null, true);
  return callback(new Error(`CORS: origin ${origin} δεν επιτρέπεται`));
}

// Rate limiting
const chatLimiter = rateLimit({ windowMs: 60 * 1000, limit: Number(process.env.CHAT_RATE_LIMIT_PER_MIN || 20), standardHeaders: true, legacyHeaders: false, message: { error: "Πολλά αιτήματα — δοκίμασε ξανά." } });
const searchLimiter = rateLimit({ windowMs: 60 * 1000, limit: Number(process.env.SEARCH_RATE_LIMIT_PER_MIN || 120), standardHeaders: true, legacyHeaders: false });
const adminLimiter = rateLimit({ windowMs: 60 * 1000, limit: Number(process.env.ADMIN_RATE_LIMIT_PER_MIN || 10), standardHeaders: true, legacyHeaders: false });
const trackLimiter = rateLimit({ windowMs: 60 * 1000, limit: Number(process.env.TRACK_RATE_LIMIT_PER_MIN || 120), standardHeaders: true, legacyHeaders: false });

const app = express();
app.set("trust proxy", 1);
app.use(cors({ origin: corsOriginCheck }));
app.use(express.json({ limit: "1mb" }));
app.use(express.static(path.join(__dirname, "../public")));

function candidatesToContext(list) {
  return list.map((p) => `- ${p.name} | Brand: ${p.brand} | Category: ${p.category} | Price: ${p.price} | URL: ${p.url}`).join("\n");
}

function buildSystemPrompt(candidates) {
  return "Είσαι ο AI Βοηθός του Cosmos Sport. Απάντα ΜΟΝΟ με βάση τα προϊόντα που δίνονται.\n\n" + candidatesToContext(candidates);
}

async function callAnthropic(system, history) {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) throw new Error("Λείπει ANTHROPIC_API_KEY");
  const resp = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json", "x-api-key": apiKey, "anthropic-version": "2023-06-01" },
    body: JSON.stringify({ model: process.env.ANTHROPIC_MODEL || "claude-sonnet-4-6", max_tokens: 1000, system, messages: history }),
  });
  if (!resp.ok) throw new Error(`API error ${resp.status}`);
  const data = await resp.json();
  return (data.content || []).filter((c) => c.type === "text").map((c) => c.text).join("\n") || "Δεν έλαβα απάντηση.";
}

async function callOpenAI(system, history) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) throw new Error("Λείπει OPENAI_API_KEY");
  const resp = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
    body: JSON.stringify({ model: process.env.OPENAI_MODEL || "gpt-4o-mini", max_tokens: 1000, messages: [{ role: "system", content: system }, ...history] }),
  });
  if (!resp.ok) throw new Error(`API error ${resp.status}`);
  const data = await resp.json();
  return data.choices?.[0]?.message?.content || "Δεν έλαβα απάντηση.";
}

async function callGemini(system, history) {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error("Λείπει GEMINI_API_KEY");
  const model = process.env.GEMINI_MODEL || "gemini-2.5-flash";
  const resp = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ systemInstruction: { parts: [{ text: system }] }, contents: history.map((m) => ({ role: m.role === "assistant" ? "model" : "user", parts: [{ text: m.content }] })) }),
  });
  if (!resp.ok) throw new Error(`API error ${resp.status}`);
  const data = await resp.json();
  return data.candidates?.[0]?.content?.parts?.map((p) => p.text).join("\n") || "Δεν έλαβα απάντηση.";
}

const PROVIDERS = { anthropic: callAnthropic, openai: callOpenAI, gemini: callGemini };

app.post("/api/chat", chatLimiter, async (req, res) => {
  try {
    const { message, history = [] } = req.body || {};
    if (!message || typeof message !== "string") return res.status(400).json({ error: "Λείπει το message" });
    if (message.length > 700) return res.status(400).json({ error: "Πολύ μεγάλο μήνυμα" });
    const candidates = EkpaSearch.search(PRODUCTS, CONCEPTS, message, 8);
    const system = buildSystemPrompt(candidates);
    const fn = PROVIDERS[PROVIDER];
    if (!fn) return res.status(500).json({ error: `Άγνωστος provider: ${PROVIDER}` });
    const fullHistory = [...history, { role: "user", content: message }];
    const reply = await fn(system, fullHistory);
    res.json({ reply, candidates_used: candidates.map((c) => c.name) });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

app.get("/api/search", searchLimiter, (req, res) => {
  const q = req.query.q || "";
  const results = EkpaSearch.search(PRODUCTS, CONCEPTS, q, 40);
  trackSearch(q, results.length);
  res.json(results);
});

app.post("/api/track-search", trackLimiter, (req, res) => {
  const { query, result_count } = req.body || {};
  trackSearch(query, Number(result_count) || 0);
  res.json({ ok: true });
});

app.post("/api/track-click", trackLimiter, (req, res) => {
  const { query, program_id } = req.body || {};
  if (!program_id) return res.status(400).json({ error: "Λείπει program_id" });
  trackClick(query, String(program_id));
  res.json({ ok: true });
});

app.get("/api/programs", searchLimiter, (req, res) => {
  res.set("Cache-Control", "public, max-age=300");
  res.json(PRODUCTS);
});

app.get("/concepts.json", searchLimiter, (req, res) => {
  res.set("Cache-Control", "public, max-age=300");
  res.json(CONCEPTS);
});

app.get("/health", (req, res) => res.json({ ok: true, products: PRODUCTS.length }));

function checkAdminToken(req, res, next) {
  const expected = process.env.ADMIN_TOKEN;
  if (!expected) return res.status(503).json({ error: "Admin δεν έχει ρυθμιστεί" });
  const provided = req.get("x-admin-token");
  if (provided !== expected) return res.status(401).json({ error: "Λάθος token" });
  next();
}

app.post("/api/admin/concepts", adminLimiter, checkAdminToken, (req, res) => {
  try {
    fs.writeFileSync(CONCEPTS_PATH, JSON.stringify(req.body, null, 2) + "\n", "utf-8");
    CONCEPTS = req.body;
    res.json({ ok: true });
  } catch (err) {
    res.status(500).json({ error: "Αποτυχία αποθήκευσης" });
  }
});

app.get("/api/admin/analytics", adminLimiter, checkAdminToken, (req, res) => {
  const entries = Object.entries(analyticsQueries);
  res.json({ tracked_queries_total: entries.length, top_queries: entries.slice(0, 100) });
});

app.listen(PORT, () => {
  console.log(`Cosmos Smart Finder listening on http://localhost:${PORT}`);
  console.log(`LLM: ${PROVIDER}`);
  console.log(`CORS: ${ALLOWED_ORIGINS.join(", ")}`);
});
