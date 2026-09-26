---
title: Which Option Fits Best
emoji: 🎯
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Pick the best option from any list, with honest percentages
models:
  - ali-rehman-ML/modern-bert-jev
---

# Which option fits best?

Give it some background, a question, and a list of options. It reads every option against
the background and tells you how likely each one is.

The same model handles 2 options or 80 — nothing is tied to a fixed list. Percentages are
calibrated, so when it says 70% it is right about 70% of the time.

Good at sorting things into known categories: bank support intents (86% out of 77 options),
voice commands (84% out of 60), customer intents (83% out of 151), legal clause types (76%
out of 100).

Bad at world knowledge: 28% on school-exam questions, barely above the 25% you would get by
guessing. It is a small model and has not memorised facts.

[Model](https://huggingface.co/ali-rehman-ML/modern-bert-jev) ·
[Code](https://github.com/ali-rehman-ML/modern-bert-jev)

---

## Deploying this

Hugging Face requires a PRO subscription to host a Gradio Space on free CPU. To deploy:

```bash
hf repos create <you>/modern-bert-jev --type space --sdk gradio --public
cp choice_model.py download_model.py space/        # the Space needs its own copies
hf upload <you>/modern-bert-jev space/ . --repo-type space
```

To run it locally or in Colab instead, see the repository README — no subscription needed.
