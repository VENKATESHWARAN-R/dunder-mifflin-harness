# Model Tier List — dunder-mifflin-harness

> **Last updated:** March 27 2026 · Source: OpenRouter API, Artificial Analysis, NVIDIA technical blog, OpenAI blog, Mistral AI blog, Z.ai

---

## Why Tiers?

This harness runs three categories of agents with very different cost/quality profiles:

| Role | Examples | What matters |
|---|---|---|
| **Tier 1 — Orchestrator** | Planner, Goal Decomposer, Architect | Max reasoning, long context, reliable tool-use. Called infrequently — cost is secondary. |
| **Tier 2 — Worker** | Coder, Tester, Debugger, Reviewer | High coding ability + context coherence. Called constantly — cost matters. |
| **Tier 3 — Router / Monitor** | Classifier, Summariser, Validator, Memory writer | Lightweight, fast, cheap. Called on *every* step. Cost is critical. |

The harness's outer loop hits Tier 3 most often. Tier 1 is the expensive rarity. Design cost budgets accordingly.

---

## Tier 1 — Orchestrator

**Use when:** Planning the full task graph, architectural decisions, multi-step reasoning with ambiguity, resolving conflicts between sub-agents.

| Model | OpenRouter ID | Cost (in/out per 1M) | Context | Strengths | Notes |
|---|---|---|---|---|---|
| **Claude Opus 4.6** ⭐ | `anthropic/claude-opus-4-6` | $5 / $25 | 1M | Best overall reasoning + agentic coherence across long sessions | First choice for orchestrator |
| **GPT-5.4 Pro** | `openai/gpt-5.4-pro` | $30 / $180 | 1M | Mandatory reasoning, 57.7% SWE-Bench Pro, computer use built-in | Use for highest-stakes planning only — very expensive |
| **Gemini 3.1 Pro Preview** | `google/gemini-3.1-pro-preview` | $2 / $12 | 1M | Strong multimodal reasoning, 90% context caching discount | Best cost-to-tier-1-quality ratio if Opus feels expensive |
| **Grok 4.20 Multi-Agent Beta** | `x-ai/grok-4.20-multi-agent-beta` | $2 / $6 | 2M | Native multi-agent design (4–16 parallel agents), deep research synthesis, real-time X data, 2M context | Purpose-built for multi-agent orchestration; beta but already useful for parallelisable planning tasks |
| **DeepSeek V3.2 Speciale** | `deepseek/deepseek-v3.2-speciale` | $0.50 / $1.20 | 164K | Ahead of GPT-5 on hard reasoning workloads per DeepSeek evals | Budget Tier 1 — context ceiling is a constraint |

**Primary recommendation:** `claude-opus-4-6` for orchestration. `grok-4.20-multi-agent-beta` when the task is inherently parallelisable (deep research, multi-branch planning). Fall back to `gemini-3.1-pro-preview` for cost control on long runs.

---

## Tier 2 — Worker

**Use when:** Writing and editing code, executing plan steps, running tools, file I/O, multi-turn implementation loops.

| Model | OpenRouter ID | Cost (in/out per 1M) | Context | Strengths | Notes |
|---|---|---|---|---|---|
| **Claude Sonnet 4.6** ⭐ | `anthropic/claude-sonnet-4-6` | $3 / $15 | 1M | Best Anthropic mid-tier, strong agentic planning + coding | Default worker |
| **GPT-5.4** | `openai/gpt-5.4` | $3 / $15 | 1M | Unifies Codex + GPT, built-in computer use, 57.7% SWE-Bench Pro | On par with Sonnet 4.6, better for GUI/browser tasks |
| **GPT-5.4 Mini** | `openai/gpt-5.4-mini` | $0.75 / $4.50 | 400K | Core GPT-5.4 capabilities at lower cost; text + image input, strong coding + tool use | High-throughput alternative to full GPT-5.4; solid for agentic coding loops at scale |
| **GLM-5 Turbo** | `z-ai/glm-5-turbo` | $1.20 / $4 | 202K | 744B MoE (40B active), deeply optimised for long-chain agentic tasks and complex instruction decomposition | Best-in-class on ZClawBench / OpenClaw agent scenarios; closed-source |
| **Gemini 3 Flash** | `google/gemini-3-flash-preview` | $0.50 / $3 | 1M | Near-Pro reasoning at Flash prices, strong tool use | Best value in this bracket |
| **DeepSeek V3.2** | `deepseek/deepseek-v3.2` | $0.26 / $0.38 | 164K | ~90% of GPT-5.4 quality at 1/50th the cost, "thinking in tool-use" | Exceptional value; context cap is the only constraint |
| **Grok 4.1 Fast** | `x-ai/grok-4.1-fast` | $0.20 / $0.50 | 2M | Fastest frontier-class model, 2M context | When full-repo context matters |

**Primary recommendation:** `deepseek/deepseek-v3.2` for volume coding work. `claude-sonnet-4-6` when coherence across long agentic sessions is non-negotiable. `gpt-5.4-mini` as a mid-cost OpenAI-aligned worker. `glm-5-turbo` for persistent/long-chain agent pipelines. `grok-4.1-fast` when you need to load an entire large codebase in one pass.

---

## Tier 3 — Router / Monitor

**Use when:** Classifying task type, routing to correct sub-agent, summarising tool output for the context store, validating outputs, writing memory entries, checking plan completeness.

| Model | OpenRouter ID | Cost (in/out per 1M) | Context | Strengths | Notes |
|---|---|---|---|---|---|
| **Gemini 3.1 Flash Lite** ⭐ | `google/gemini-3.1-flash-lite-preview` | $0.25 / $1.50 | 1M | 2.5x faster TTFT than 2.5 Flash, 1M context | Best all-round Tier 3 |
| **GPT-5.4 Nano** | `openai/gpt-5.4-nano` | $0.20 / $1.25 | 400K | Fastest/cheapest GPT-5.4 family; text + image; optimised for classification, extraction, ranking, sub-agent dispatch | Excellent drop-in for high-volume routing where OpenAI tool-calling format is required |
| **Mistral Small 4** | `mistralai/mistral-small-2603` | $0.15 / $0.60 | 256K | 119B MoE (6B active), multimodal text+image, configurable reasoning, 53.6% SWE-bench, Apache 2.0 | Cheapest model in this bracket with genuine multimodal + coding capacity; great validator/summariser |
| **Claude Haiku 4.5** | `anthropic/claude-haiku-4-5` | $1 / $5 | 200K | Matches Sonnet 4 quality at 1/3 cost, extended thinking optional | When you need Anthropic-aligned output at Tier 3 |
| **ByteDance Seed 1.6 Flash** | `bytedance/seed-1.6-flash` | $0.07 / $0.30 | 256K | Ultra-fast multimodal, best raw speed in class | Near-free monitoring at high volume |
| **Qwen3.5 Plus** | `qwen/qwen3.5-plus` | $0.40 / $2 | 1M | Strong reasoning at ultra-low cost | Good fallback |

**Primary recommendation:** `gemini-3.1-flash-lite-preview` as default monitor/router. `gpt-5.4-nano` when the pipeline is OpenAI-native and needs image support. `mistral-small-2603` for cost-efficient multimodal validation. `seed-1.6-flash` when cost is the only constraint and volume is very high.

---

## Open-Source Only Tier List

For self-hosted deployments, air-gapped environments, or cost-zero experimentation.
All models below are either fully free on OpenRouter or carry open weights you can host yourself.

### OSS Tier 1 — Orchestrator

| Model | OpenRouter ID | Cost on OR | Context | Why |
|---|---|---|---|---|
| **Nemotron 3 Super** ⭐ | `nvidia/nemotron-3-super-120b-a12b` | $0.10 / $0.50 (free tier: $0) | 1M native (OR serves 262K) | 85.6% PinchBench (best open model in class), 60.47% SWE-Bench Verified, 478 tok/sec. Built for multi-agent systems. NVIDIA open license. |
| **Kimi K2.5** | `moonshotai/kimi-k2.5` | $0.50 / $2 | 256K | Open-source, native multi-agent with 100 sub-agents + 1,500 parallel tool calls. New agentic paradigm. |
| **DeepSeek V3.2 Speciale** | `deepseek/deepseek-v3.2-speciale` | $0.50 / $1.20 | 164K | MIT-adjacent, strong reasoning, agentic tool synthesis pipeline. Self-hostable (weights released). |

**Notes on Nemotron 3 Super:** OpenRouter currently serves 262K context — the 1M window is available on self-hosted NIM deployments. For production harnesses that need full 1M, deploy via NVIDIA NIM on Blackwell hardware. Also note: the model is verbose (15x more tokens than average in benchmarks) — factor this into token budget planning.

### OSS Tier 2 — Worker

| Model | OpenRouter ID | Cost on OR | Context | Why |
|---|---|---|---|---|
| **Xiaomi MiMo-V2-Flash** ⭐ | `xiaomi/mimo-v2-flash` | Free | 256K | #1 open-source on SWE-bench, 309B MoE with hybrid thinking. Zero cost on OR. |
| **GPT-OSS-120b** | `openai/gpt-oss-120b:free` | Free | 131K | 117B MoE (5.1B active), Apache 2.0. 62.4% SWE-bench Verified, 96.6% AIME 2024 (w/ tools), native tool use, configurable reasoning depth. Runs on single H100 with MXFP4. |
| **GPT-OSS-20b** | `openai/gpt-oss-20b:free` | Free | 131K | 21B MoE (3.6B active), Apache 2.0. 98.7% AIME 2025, 71.5% GPQA Diamond, fine-tuneable. Runs on 16GB VRAM — excellent edge/local deployment model. |
| **Mistral Small 4** | `mistralai/mistral-small-2603` | $0.15 / $0.60 | 256K | 119B MoE (6B active), Apache 2.0. Multimodal (text + image), 53.6% SWE-bench Verified, 40% faster than Mistral Small 3. Combines Magistral reasoning + Pixtral vision + Devstral coding in one model. |
| **Devstral 2** | `mistral/devstral-2-2512` | Free / $0.05 / $0.22 | 256K | Mistral's 123B agentic coder. Multi-file orchestration, failure recovery. Modified MIT. |
| **Nemotron 3 Super (paid)** | `nvidia/nemotron-3-super-120b-a12b` | $0.10 / $0.50 | 262K | Use paid tier for guaranteed throughput SLAs in production. |
| **DeepSeek V3.2** | `deepseek/deepseek-v3.2` | $0.26 / $0.38 | 164K | Open weights (MIT), best cost/quality in any tier. Self-hostable. |

### OSS Tier 3 — Router / Monitor

| Model | OpenRouter ID | Cost on OR | Context | Why |
|---|---|---|---|---|
| **Nemotron 3 Nano** ⭐ | `nvidia/nemotron-3-nano-30b-a3b:free` | Free | 262K | 30B MoE, fully open weights. Purpose-built for agentic subtasks. |
| **GPT-OSS-20b** | `openai/gpt-oss-20b:free` | Free | 131K | Also viable as a monitor/validator. 16GB-deployable, strong function calling and CoT. Use when you want GPT-format tool calling at zero cost on OR. |
| **Mistral Small 4** | `mistralai/mistral-small-2603` | $0.15 / $0.60 | 256K | Apache 2.0, multimodal, configurable reasoning. At $0.15/1M input it's viable for high-volume validation passes. Best choice when image + reasoning in the same validator call is needed. |
| **Step 3.5 Flash** | `stepfun/step-3.5-flash` | Free | 256K | 11B active of 196B MoE, very fast, free tier available. Good general routing. |
| **DeepSeek V3.1 Nex-N1** | `deepseek/v3.1-nex-n1` | Free | 131K | Post-trained for agent autonomy and tool use. Strong lightweight classifier. |

---

## Recommended Configurations

### Configuration A: Fully Open-Source (zero API cost)
```
Tier 1 (Orchestrator):  nvidia/nemotron-3-super-120b-a12b:free
Tier 2 (Worker):        xiaomi/mimo-v2-flash  →  mistral/devstral-2-2512 (fallback)
Tier 3 (Router):        nvidia/nemotron-3-nano-30b-a3b:free
```
**Estimated cost per complex task:** ~$0 (rate-limited) or ~$0.50–2 at paid tiers

### Configuration B: Best Quality / Cost Balance
```
Tier 1 (Orchestrator):  google/gemini-3.1-pro-preview
Tier 2 (Worker):        deepseek/deepseek-v3.2
Tier 3 (Router):        google/gemini-3.1-flash-lite-preview
```
**Estimated cost per complex task:** ~$0.50–3

### Configuration C: Max Quality (Anthropic-first)
```
Tier 1 (Orchestrator):  anthropic/claude-opus-4-6
Tier 2 (Worker):        anthropic/claude-sonnet-4-6
Tier 3 (Router):        anthropic/claude-haiku-4-5
```
**Estimated cost per complex task:** ~$5–50+

---

*Pricing sourced from OpenRouter API, teamday.ai (March 6 2026), Artificial Analysis, and NVIDIA technical blog (March 11 2026). Prices may change — pin model versions in production configs.*