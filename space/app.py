"""Interactive demo for the modern-bert-jev choice scorer."""
import json

import gradio as gr
import torch
from huggingface_hub import hf_hub_download

from choice_model import ChoiceScorer, chunks, encode, tokenizer
from download_model import fetch_backbone

ADAPTER = "ali-rehman-ML/modern-bert-jev"
BACKBONE = "answerdotai/ModernBERT-base"
MAX_OPTIONS = 80

fetch_backbone(BACKBONE)
settings = json.loads(open(hf_hub_download(ADAPTER, "config.json")).read())
TEMPERATURE = json.loads(open(hf_hub_download(ADAPTER, "calibration.json")).read())["temperature"]
tok = tokenizer(BACKBONE)
model = ChoiceScorer(settings["lora_rank"], settings["lora_alpha"], "cpu", BACKBONE)
model.load_adapter(hf_hub_download(ADAPTER, "adapter.pt"))
model.eval()


def read_background(text):
    """Plain text is used as-is; pasted JSON is passed through as structure."""
    text = (text or "").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    return parsed if isinstance(parsed, (dict, list)) else text


def score(background, question, options):
    question = (question or "").strip()
    if not question:
        raise gr.Error("Please write a question.")
    choices = {}
    for line in (options or "").splitlines():
        line = line.strip()
        if line and line not in choices:
            choices[line] = line
    if len(choices) < 2:
        raise gr.Error("Please list at least two different options, one per line.")
    if len(choices) > MAX_OPTIONS:
        raise gr.Error(f"This demo runs on a free CPU, so it stops at {MAX_OPTIONS} options. "
                       f"You gave {len(choices)}.")

    record = {"context": read_background(background), "question": question, "choices": choices}
    tokens, truncated = encode(record, tok, settings["max_length"])
    with torch.inference_mode():
        scores = torch.cat([model(batch) for batch in chunks(tokens, 16, "cpu")])
    ranked = dict(zip(choices, (scores.float() / TEMPERATURE).softmax(0).tolist()))

    best = max(ranked, key=ranked.get)
    note = f"### {best}\n**{ranked[best]:.0%} sure**, chosen from {len(choices)} options."
    if truncated:
        note += "\n\nYour background text was long, so the end of it was trimmed."
    return ranked, note


EMOTIONS = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
            "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
            "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
            "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"]

BANKING = ["Unable to verify identity", "Activate my card", "Age limit", "Apple pay or google pay",
           "Atm support", "Card arrival", "Card not working", "Change pin", "Lost or stolen card",
           "Pending card payment", "Top up by card", "Verify my identity", "Why verify identity"]

VOICE = ["Alarm set", "Calendar query", "Cooking recipe", "Datetime query", "Email send",
         "Music play", "News query", "Recommendation events", "Recommendation locations",
         "Transport query", "Weather query"]

LEGAL = ["Adjustments", "Amendments", "Arbitration", "Assignments", "Confidentiality",
         "Counterparts", "Expenses", "Governing Laws", "Indemnifications", "Notices",
         "Severability", "Terminations", "Waivers", "Warranties"]

EXAMPLES = [
    ["How long will it take for my ID to verify?",
     "Which banking support intent does the customer's message express?",
     "\n".join(BANKING)],
    ["I did that in origins but now that theres mercenaries and its sooo annoying to deal with "
     "them so i stay as quiet as i can until im caught.",
     "Which emotion does the comment primarily express?",
     "\n".join(EMOTIONS)],
    ["are there any free events on in my area today",
     "Which intent does the user's request to a voice assistant express?",
     "\n".join(VOICE)],
    ["Any notices or demands required or contemplated hereunder shall be written and shall be "
     "effective two days after the placing thereof in the United States mails postage prepaid "
     "or with a nationally-recognized courier service such as Federal Express, addressed to the "
     "relevant party at its address set forth below.",
     "Which contract provision category does this clause belong to?",
     "\n".join(LEGAL)],
    [json.dumps({"premise": "All the customers received their refunds yesterday.",
                 "hypothesis": "Some customers have not received their refunds."}, indent=2),
     "What is the relationship of `hypothesis` to `premise`?",
     "The hypothesis is definitely true given the premise.\n"
     "The hypothesis might be true; the premise does not settle it.\n"
     "The hypothesis is definitely false given the premise."],
]

INTRO = """
# Which option fits best?

Give it some **background**, a **question**, and a **list of options**. It reads every option
against the background and tells you how likely each one is.

It works the same whether you give it 2 options or 80 — nothing is fixed to a particular list.
The percentages are tuned to be honest: when it says 70%, it is right about 70% of the time.

*Click an example below to see it in action.*
"""

DETAIL = """
This is a small model — **150 million** settings, of which only **1.6 million** were trained.
It is `ModernBERT-base` with a thin adapter on top, trained on nine different option-picking
tasks at once: bank support messages, voice assistant commands, emotions in comments, legal
contract clauses, and sentence logic.

**What it is good at:** sorting things into a known list of categories. Bank support intents
(86% right out of 77 options), voice commands (84% out of 60), customer intents (83% out of 151),
legal clause types (76% out of 100).

**What it is bad at:** anything needing world knowledge. On school-exam questions it scores
28%, barely better than guessing at 25%. It is a small model and it has not memorised facts.
Do not use it for trivia or exam questions.

**Speed:** this demo runs on a free processor, not a graphics card, so a long list of options
takes a few seconds. It scores each option separately, so 80 options means 80 passes.

[Model files](https://huggingface.co/ali-rehman-ML/modern-bert-jev) ·
[Code](https://github.com/ali-rehman-ML/modern-bert-jev) · Apache-2.0
"""

with gr.Blocks(title="Which option fits best?", theme=gr.themes.Soft()) as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column():
            background = gr.Textbox(label="Background", lines=5,
                                    placeholder="The text to think about.")
            question = gr.Textbox(label="Question", lines=2,
                                  placeholder="What do you want to know about it?")
            options = gr.Textbox(label="Options — one per line", lines=8,
                                 placeholder="First option\nSecond option\nThird option")
            button = gr.Button("Score the options", variant="primary")
        with gr.Column():
            answer = gr.Markdown()
            chart = gr.Label(label="How likely each option is", num_top_classes=8)
    gr.Examples(EXAMPLES, inputs=[background, question, options],
                label="Try one of these", cache_examples=False)
    with gr.Accordion("What is this, and what is it bad at?", open=False):
        gr.Markdown(DETAIL)
    button.click(score, [background, question, options], [chart, answer])

if __name__ == "__main__":
    demo.launch()
