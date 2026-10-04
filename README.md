# Build a Transformer

A local browser game in English, with a balanced default interface for a monitor or projector. No dependencies, accounts, or external services are required.

## Open the game

Open `index.html` in a modern browser. On macOS, double-click `Open game.command`.

Alternatively, serve this folder locally:

```sh
python3 -m http.server 18421 --bind 127.0.0.1 --directory /path/to/transformer-lab
```

Then open http://127.0.0.1:18421. File and HTTP addresses have separate saved progress. The game works without localStorage if the browser restricts it, but progress may not persist.

## Play

1. Choose BERT, GPT-2, or Attention Is All You Need. All three levels are available immediately.
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

Normal mode requires the original model’s normalization order and FFN. Experiment mode permits alternative normalization order, RMSNorm, other activations, and FFNs up to 16 operations. It checks widths, nonlinearities, masks, and attention sources. Experimental success does not award original model completion or imply training quality.

The palette includes extra choices such as BatchNorm, RoPE, Sigmoid, and Tanh. RoPE acts on Q/K, rather than providing embedding vectors to add. A single SiLU operation does not implement SwiGLU, which requires a gated branch.

## Normalization

The original Transformer and BERT use Post-LN: `y = LayerNorm(x + F(x))`. This is a valid architecture.

GPT-2 uses LayerNorm before attention and FFN, plus final LayerNorm: `y = x + F(LayerNorm(x))`. Many modern GPT-style models use pre-norm because the direct residual path often helps deep-stack optimization. LayerNorm versus RMSNorm describes the type; pre/post describes the placement.

The game shows the inference structure. Dropout, padding masks, embedding scaling by √d_model, and BERT pre-training heads are omitted. Full Attention permits the full valid context, not attention to padding.

## Papers and original figures

**Paper & figures** opens original publication links, embedded images, zoom, and PNG downloads. The eight images work offline and match the PNG files in `assets/`. `assets/SOURCES.json` records source URLs, pages, and checksums.

- [Attention Is All You Need, 2017](https://arxiv.org/abs/1706.03762): Figure 1 (architecture, page 3), Figure 2 (attention, page 4).
- [BERT, 2018](https://arxiv.org/abs/1810.04805): Figure 1 (pre-training and fine-tuning, page 3), Figure 2 (embeddings, page 5), Figure 3 (BERT/GPT/ELMo, page 13). That comparison depicts GPT-1.
- [GPT-2, 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf): section 2.3 (Pre-LN excerpt, page 4), Table 2 (model sizes, page 4), Figure 2 (CBT results, page 5). The report has no detailed Transformer block diagram; the game follows the report and original code.
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
