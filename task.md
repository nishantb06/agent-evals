# Ollive: AI Evals Platform Challenge

## Overview

Build an Evals platform and evaluate **two AI Agents** designed to assist users in making better decisions around health and lifestyle.

## AI Agent Specification

**Build a Wellness Assistant** that helps users with health and lifestyle choices.  
You may create your own knowledge bank or use this resource to save time:  
[Knowledge Bank (Google Drive Link)](https://drive.google.com/file/d/1Jbf-LPSlojajZ2-GhL3omEr0TNrfB0lL/view?usp=drive_link)

### Agent Variants

1. **Open Source Assistant**
   - Deploy one version using an open-source model (e.g., from Hugging Face).
   - Examples:
     - Qwen 2.5
     - Llama 3.2
     - Phi-3
     - Mistral
     - Any equivalent OSS model

2. **Frontier Model Assistant**
   - Deploy the same agent experience using a foundation model from a provider.
   - Examples:
     - Claude (any)
     - GPT (any)
     - Gemini (any)
     - DeepSeek (any)
     - Grok (any)
     - Any equivalent

### Core Assistant Features

- Multi-turn conversations
- Short-term conversational memory/context
- 2 tool calls: `lookup_kb`, `search_web`

### Interface Options

You may use any lightweight setup:
- Streamlit
- Gradio
- FastAPI
- Telegram/Discord bot
- CLI
- Web app
- *(or any equivalent interface)*

---

## Evaluation

**Design an Evals Platform** to test and compare both assistants on three axes:

1. **Hallucination**
2. **Bias & Harmful Outputs**
    - Stereotypes
    - Discriminatory behavior
    - Unsafe responses
3. **Content Safety**
    - Jailbreak resistance
    - Refusal handling
    - Robustness to harmful prompts

*You will run your evals platform on a dataset to assess the judge's quality.*

---

## Deliverables

1. **GitHub Repository**  
   Complete source code.

2. **README**
   - Setup instructions
   - Architecture decisions
   - Tradeoffs made
   - What you would improve with more time

3. **Short Evaluation Report (1 page)**
   - Comparison results (as infographics)
   - Recommendations

4. **Demo (Optional)**
   - Hosted link, screenshots, or Loom video

---

## Bonus

- Add guardrails/safety based on the eval results
- Deploy the OSS model publicly
- Provide a cost + latency table for the OSS deployment

---

## Submission

Please send:
- GitHub repo
- Evaluation PDF
- Demo link (optional)

---

Looking forward to seeing what you build! 