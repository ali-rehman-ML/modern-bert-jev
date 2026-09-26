import { useEffect, useRef, useState } from "react";
import { EXAMPLES } from "./examples.js";
import { score, wake } from "./model.js";

const TEMPERATURE = 1.2023;
const MAX_OPTIONS = 60;

function Bar({ option, probability, leader }) {
  return (
    <li className={leader ? "bar bar--leader" : "bar"}>
      <div className="bar__label">
        <span className="bar__option">{option}</span>
        <span className="bar__value">{(probability * 100).toFixed(1)}%</span>
      </div>
      <div className="bar__track">
        <div className="bar__fill" style={{ width: `${Math.max(probability * 100, 0.6)}%` }} />
      </div>
    </li>
  );
}

export default function App() {
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState(null);
  const [outcome, setOutcome] = useState(null);
  const [background, setBackground] = useState(EXAMPLES[0].background);
  const [question, setQuestion] = useState(EXAMPLES[0].question);
  const [options, setOptions] = useState(EXAMPLES[0].options.join("\n"));
  const [calibrated, setCalibrated] = useState(true);
  const resultsRef = useRef(null);

  useEffect(() => {
    document.title = "Which option fits best?";
  }, []);

  function pick(example) {
    setBackground(example.background);
    setQuestion(example.question);
    setOptions(example.options.join("\n"));
    setOutcome(null);
    setError(null);
  }

  async function warm() {
    setError(null);
    setStatus("loading");
    try {
      await wake();
      setStatus("ready");
    } catch (failure) {
      setError(failure.message);
      setStatus("idle");
    }
  }

  async function run() {
    const list = [];
    const seen = new Set();
    for (const line of options.split("\n")) {
      const trimmed = line.trim();
      if (trimmed && !seen.has(trimmed)) {
        seen.add(trimmed);
        list.push(trimmed);
      }
    }
    if (!question.trim()) return setError("Write a question first.");
    if (list.length < 2) return setError("List at least two different options, one per line.");
    if (list.length > MAX_OPTIONS) {
      return setError(`This runs in your browser, so it stops at ${MAX_OPTIONS} options. You gave ${list.length}.`);
    }

    setError(null);
    setStatus("scoring");
    try {
      const result = await score({
        background,
        question,
        options: list,
        temperature: calibrated ? TEMPERATURE : 1,
      });
      setOutcome(result);
      setStatus("ready");
      requestAnimationFrame(() => resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" }));
    } catch (failure) {
      setError(failure.message);
      setStatus("idle");
    }
  }

  const busy = status === "loading" || status === "scoring";

  return (
    <div className="page">
      <header className="hero">
        <h1>Which option fits best?</h1>
        <p className="hero__lead">
          Give it some background, a question, and a list of options. It reads every option and
          tells you how likely each one is. Two options or sixty &mdash; same model either way.
        </p>
        <p className="hero__note">
          Free, no sign-up. The model wakes on demand and sleeps when nobody is using it.
        </p>
      </header>

      <div className="examples">
        {EXAMPLES.map((example) => (
          <button key={example.label} className="chip" onClick={() => pick(example)} type="button">
            <span className="chip__label">{example.label}</span>
            <span className="chip__hint">{example.hint}</span>
          </button>
        ))}
      </div>

      <main className="panels">
        <section className="panel">
          <label className="field">
            <span className="field__name">Background</span>
            <textarea rows={5} value={background} onChange={(event) => setBackground(event.target.value)}
                      placeholder="The text to think about." />
          </label>
          <label className="field">
            <span className="field__name">Question</span>
            <textarea rows={2} value={question} onChange={(event) => setQuestion(event.target.value)}
                      placeholder="What do you want to know about it?" />
          </label>
          <label className="field">
            <span className="field__name">
              Options <span className="field__aside">one per line</span>
            </span>
            <textarea rows={8} value={options} onChange={(event) => setOptions(event.target.value)}
                      placeholder={"First option\nSecond option\nThird option"} />
          </label>

          <label className="toggle">
            <input type="checkbox" checked={calibrated} onChange={(event) => setCalibrated(event.target.checked)} />
            <span>
              Calibrate the percentages
              <em>Divides the scores by {TEMPERATURE}. Turn it off to see the raw, overconfident numbers.</em>
            </span>
          </label>

          <button className="run" onClick={run} disabled={busy} type="button">
            {status === "loading" ? "Waking the server…" : status === "scoring" ? "Scoring…" : "Score the options"}
          </button>

          {busy && (
            <p className="hint">
              The server sleeps when idle, so the first request after a quiet spell takes
              about ten seconds to wake. After that it answers in under a second.
            </p>
          )}
          {error && <p className="error">{error}</p>}
        </section>

        <section className="panel panel--results" ref={resultsRef}>
          {!outcome && !busy && (
            <div className="empty">
              <p>Results appear here.</p>
              <button className="ghost" onClick={warm} type="button">
                Wake the server
              </button>
            </div>
          )}
          {outcome && (
            <>
              <div className="verdict">
                <span className="verdict__eyebrow">Best match</span>
                <strong>{outcome.results[0].option}</strong>
                <span className="verdict__confidence">
                  {(outcome.results[0].probability * 100).toFixed(0)}% sure, from{" "}
                  {outcome.results.length} options
                </span>
              </div>
              <ul className="bars">
                {outcome.results.slice(0, 10).map((entry, index) => (
                  <Bar key={entry.option} {...entry} leader={index === 0} />
                ))}
              </ul>
              <p className="meta">
                {outcome.results.length} options scored in {outcome.milliseconds} ms
                {outcome.results.length > 10 && ` · showing the top 10`}
              </p>
            </>
          )}
        </section>
      </main>

      <footer className="footer">
        <nav className="links">
          <a href="https://huggingface.co/ali-rehman-ML/modern-bert-jev">Model</a>
          <a href="https://github.com/ali-rehman-ML/modern-bert-jev">Code</a>
          <span>Apache-2.0</span>
        </nav>
      </footer>
    </div>
  );
}
