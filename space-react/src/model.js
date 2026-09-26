const ENDPOINT = "https://ali-r-emblem--modern-bert-jev-api-api-web.modal.run";

/** Plain text stays text; pasted JSON is sent as structure, matching training. */
function readBackground(text) {
  const trimmed = (text || "").trim();
  try {
    const parsed = JSON.parse(trimmed);
    return parsed && typeof parsed === "object" ? parsed : trimmed;
  } catch {
    return trimmed;
  }
}

async function call(path, body, signal) {
  const response = await fetch(`${ENDPOINT}${path}`, {
    method: body ? "POST" : "GET",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
    signal,
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail || `The server said ${response.status}.`);
  }
  return response.json();
}

/**
 * The server sleeps when nobody is using it, so the first request after a quiet
 * spell pays a cold start. Pinging health lets the UI say so instead of looking
 * frozen.
 */
export function wake(signal) {
  return call("/health", null, signal);
}

export async function score({ background, question, options, temperature }) {
  const started = performance.now();
  const payload = await call("/predict", {
    context: readBackground(background),
    question: question.trim(),
    choices: Object.fromEntries(options.map((option) => [option, option])),
    temperature,
  });

  return {
    results: options
      .map((option) => ({
        option,
        probability: payload.probabilities[option] ?? 0,
        score: payload.scores[option] ?? 0,
      }))
      .sort((a, b) => b.probability - a.probability),
    milliseconds: Math.round(performance.now() - started),
    tokens: payload.tokens,
  };
}
