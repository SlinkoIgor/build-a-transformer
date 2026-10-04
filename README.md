# Build a Transformer

A static browser game in English, with a balanced default interface for a monitor or projector. No dependencies, accounts, or external services are required.

Play online: **https://slinkoigor.github.io/build-a-transformer/**

## Open the game

Open `index.html` in a modern browser. On macOS, double-click `Open game.command`.

Alternatively, serve this folder locally:

```sh
python3 -m http.server 18421 --bind 127.0.0.1 --directory /path/to/transformer-lab
```

Then open http://127.0.0.1:18421. File and HTTP addresses have separate saved progress. The game works without localStorage if the browser restricts it, but progress may not persist.

## Play

1. Choose BERT, GPT-2, Attention Is All You Need, or Qwen3.8-27B. All four levels are available immediately.
2. Start with **Architecture**. The entire network is on one canvas. Blocks form a vertical sequence, with flow **from bottom to top**, as in the original paper figure. Empty slots all say **Choose a block**, without revealing what belongs there. The original Transformer shows its encoder and decoder side by side, connected by the encoder K/V memory path.
3. Choose a block, then click a slot. You can also choose the slot first or drag and drop. Blocks can be reused.
4. **Check architecture** verifies the outer modules. Once they are correct, **Inside blocks** unlocks. Both views use the complete network canvas, without Input/Block/Output step tabs. Architecture alone does not complete the model.
5. Build the individual operations inside each module. Attention contains Q/K/V projections, head operations, masks, and softmax. Cross-attention has separate Q, K, and V source selectors. Feed Forward contains editable layers and activations.
6. The progress arrow beside a module opens its internal operations. **Add & Norm** has its own workbench. Pre-LN shows separate **Normalization** and **Residual Add** modules, before and after the sublayer respectively. Each also opens into an editable operation. The side path carries the original input to residual addition. Inside blocks expands input embeddings and output operations directly on the canvas.
7. **Check build** verifies both stages and explains errors; click an explanation to find the relevant slot. **Trace shapes** follows the data flow through one representative block. It does not compute neural network weights.

Return to Architecture at any time. Removing or replacing an outer module preserves its internal operations. Progress from earlier versions is migrated automatically, without discarding existing work. Older combined Pre-LN wrappers become separate normalization and addition modules.

The default **Standard** text size uses 17px block and slot controls, with smaller secondary labels and compact spacing. **Settings → Text size: Large** gives 22px controls when needed. The compact header and toolbar leave room for the network. **Fullscreen** gives the build more room. Smaller screens can scroll the build area.

**Settings** contains dimensions, normalization order, experiment mode, reference build, reset, and JSON export. Changes save automatically when storage is available. `Esc` closes dialogs or deselects; `⌘/Ctrl + Z` undoes a change; `/` focuses block search.

## Reference architectures

| Level | Year | Attention | Normalization | FFN | Positions |
|---|---|---|---|---|---|
| BERT-base | 2018 | Bidirectional self | Post-LN | Linear → GELU → Linear | Learned + token type |
| GPT-2 small | 2019 | Causal self | Pre-LN + final LayerNorm | Linear → GELU → Linear | Learned |
| Attention Is All You Need · Transformer base | 2017 | Encoder self, decoder causal self, decoder cross | Post-LN | Linear → ReLU → Linear | Sinusoidal |
| Qwen3.8-27B · text backbone | 2026 | Gated DeltaNet + gated causal GQA | Pre-RMSNorm + final RMSNorm | SwiGLU: two projections → SiLU gate × value → down projection | Partial RoPE on attention Q/K |

Normal mode requires the original model’s normalization order and FFN. Experiment mode permits alternative normalization order, RMSNorm, other activations, and FFNs up to 16 operations. It checks widths, nonlinearities, masks, and attention sources. Experimental success does not award original model completion or imply training quality.

The palette includes extra choices such as BatchNorm, RoPE, Sigmoid, and Tanh. RoPE acts on Q/K, rather than providing embedding vectors to add. Qwen’s SwiGLU workbench provides independent gate and value projections, SiLU, multiplication, and the down projection. A single SiLU operation does not implement SwiGLU.

## Qwen3.8-27B

The newest official Qwen checkpoint in the requested 27–30B range, checked on October 4, 2026, is **Qwen3.8-27B**. Its model repository was created on August 5, 2026. It inherits the Qwen3.5 hybrid architecture; it is a dense model, with no mixture of experts in this size.

This level builds the **text backbone**. The full checkpoint also contains a vision encoder. Vision/video processing and training-only multi-token prediction are omitted. The diagram shows one representative DeltaNet layer repeated three times, followed by one gated-attention layer, with the group repeated sixteen times: **48 DeltaNet + 16 attention = 64 layers**. Repetition represents distinct layers with separate learned parameters.

- Model width 5120; SwiGLU hidden width 17408; vocabulary size 248320.
- Full attention: 24 Q heads, 4 K/V heads, explicit head width 256. Q/K have per-head RMSNorm; partial RoPE rotates 64 of 256 features. GQA shares each K/V head across six Q heads. Concatenation produces 6144 features, projected back to 5120.
- The checkpoint sets `output_gate_type: "swish"`. The game follows this setting. The older inherited Transformers implementation linked in Sources still uses a sigmoid gate; it is not used to override the checkpoint-specific activation. The fused Q/gate projection is split into two branches in the game for teaching.
- DeltaNet: 16 Q/K heads and 48 V heads, head width 128, causal depthwise convolution of width 4, SiLU, Q/K L2 normalization, query scaling, sigmoid update strength, and learned exponential decay. The recurrent matrix state is 128×128 per value head. The gate applies SiLU to its own projection after output RMSNorm. DeltaNet has no softmax attention matrix or RoPE.
- Both layer types have RMSNorm before the mixer and SwiGLU; residual Add follows each. A final RMSNorm precedes the untied language model head.
- No additive position vectors are used. The input module contains token embeddings only.

The game validates every branch operation and SwiGLU width. Delta memory update/read and learned decay are compound mathematical building blocks; the game illustrates structure and tensor flow, not numerical inference. Dimension changes require Experiment mode for this profile. The checkpoint’s mixer and SwiGLU operations remain prescribed in that mode; the serial deeper-FFN experiments are available in the other three profiles.

The official model card and configuration are linked under Papers, with local source documents for offline use. There is no fabricated “paper figure” for Qwen.

## Normalization

The original Transformer and BERT use Post-LN: `y = LayerNorm(x + F(x))`. This is a valid architecture.

GPT-2 uses LayerNorm before attention and FFN, plus final LayerNorm: `y = x + F(LayerNorm(x))`. Many modern GPT-style models use pre-norm because the direct residual path often helps deep-stack optimization. LayerNorm versus RMSNorm describes the type; pre/post describes the placement.

The game shows the inference structure. Dropout, padding masks, embedding scaling by √d_model, and BERT pre-training heads are omitted. Full Attention permits the full valid context, not attention to padding.

## Papers and original figures

**Paper & figures** opens original publication links, embedded images, zoom, and PNG downloads. The eight original paper images work offline and match the PNG files in `assets/`. `assets/SOURCES.json` records source URLs, pages, and checksums.

- [Attention Is All You Need, 2017](https://arxiv.org/abs/1706.03762): Figure 1 (architecture, page 3), Figure 2 (attention, page 4).
- [BERT, 2018](https://arxiv.org/abs/1810.04805): Figure 1 (pre-training and fine-tuning, page 3), Figure 2 (embeddings, page 5), Figure 3 (BERT/GPT/ELMo, page 13). That comparison depicts GPT-1.
- [GPT-2, 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf): section 2.3 (Pre-LN excerpt, page 4), Table 2 (model sizes, page 4), Figure 2 (CBT results, page 5). The report has no detailed Transformer block diagram; the game follows the report and original code.
- [Qwen3.8-27B official model card](https://huggingface.co/Qwen/Qwen3.8-27B) and [pinned configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json). Local copies/excerpts are in `assets/`.
- [Qwen3.5-family implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py): shared architectural family, with the checkpoint-specific gate caveat above.
- [Original GPT-2 code: ln_1, ln_2, ln_f](https://github.com/openai/gpt-2/blob/master/src/model.py)
- [Original BERT code](https://github.com/google-research/bert/blob/master/modeling.py)
- [On Layer Normalization in the Transformer Architecture, 2020](https://arxiv.org/abs/2002.04745)
- [RMSNorm, 2019](https://arxiv.org/abs/1910.07467)
- [Llama, 2023](https://arxiv.org/abs/2302.13971)

The user-supplied reference image is preserved separately as `reference.png`.

## Files

- `index.html`: entry point.
- `engine.js`: model profiles, structure, validation, and tensor routes.
- `app.js`: interface and interaction.
- `styles.css`, `network.css`, `workbench.css`, `papers.css`: network, responsive, and dialog layouts.
- `papers.js`, `paper-images.js`, `assets/`: original sources and offline figures.

All scripts run directly in the browser; no build step, npm, or CDN is needed.

## Development and checks

No build step is required. The pure engine checks use Node.js:

```sh
node tests/engine.cjs
node tests/structure.cjs
node tests/qwen.cjs
```

The Qwen browser checks use Python with Playwright (`pip install playwright`, then `playwright install chromium`):

```sh
python tests/qwen-ui.py
```

Set `GAME_URL` to test another local server or the published Pages URL. Set `CHROME_EXECUTABLE` to use an existing Chrome installation. Screenshots go to ignored `test-results/` (override with `TEST_ARTIFACTS`). Tests use an isolated browser context and do not change the player’s saved progress.

## GitHub Pages

GitHub Pages publishes the root of the `main` branch, using `.nojekyll`. Push a commit to `main` to update the site. All asset URLs are relative, so the game works under the repository path without a bundler or external CDN.

The original accepted three-model version is preserved in commit `d849ed1` and tag `v0.9-before-qwen`.
