(() => {
  'use strict';
  const figure = (id, title, page, description) => ({ id, title, page, description, file: 'assets/' + id + '.png' });
  window.TransformerPapers = {
    bert: {
      model: 'BERT-base', title: 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
      authors: 'Devlin, Chang, Lee & Toutanova', year: '2018 / NAACL 2019',
      url: 'https://arxiv.org/abs/1810.04805', pdf: 'https://arxiv.org/pdf/1810.04805',
      code: 'https://github.com/google-research/bert/blob/master/modeling.py',
      norm: 'Post-LN', normSummary: 'LayerNorm follows each residual addition. Input embeddings also have their own LayerNorm; there is no extra final LayerNorm after the stack.',
      normEvidence: 'Section 3 describes BERT as a bidirectional Transformer encoder based on Vaswani et al. The original modeling.py uses layer_norm(attention_output + layer_input), then layer_norm(layer_output + attention_output).',
      formula: 'x = LayerNorm₁(x + SelfAttention(x))\nx = LayerNorm₂(x + FFN(x))',
      facts: ['12 encoder blocks · 12 heads · d_model = 768 · d_ff = 3072', 'Bidirectional self-attention; no cross-attention.', 'Learned token, segment, and position embeddings; GELU in the FFN.'],
      figures: [
        figure('bert-figure-3', 'Figure 3 · BERT bidirectional Transformer', 13, 'The original comparison of BERT, OpenAI GPT, and ELMo. OpenAI GPT in this figure is GPT-1, not GPT-2.'),
        figure('bert-figure-2', 'Figure 2 · Input embeddings', 5, 'Token embeddings + segment embeddings + position embeddings.'),
        figure('bert-figure-1', 'Figure 1 · Pre-training and fine-tuning', 3, 'A shared BERT encoder with different output heads. The game builds the encoder; MLM, NSP, and downstream task heads are omitted.'),
      ],
    },
    gpt: {
      model: 'GPT-2 small', title: 'Language Models are Unsupervised Multitask Learners',
      authors: 'Radford, Wu, Child, Luan, Amodei & Sutskever', year: '2019',
      url: 'https://openai.com/index/better-language-models/',
      pdf: 'https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf',
      code: 'https://github.com/openai/gpt-2/blob/master/src/model.py',
      norm: 'Pre-LN + final LayerNorm', normSummary: 'LayerNorm precedes masked self-attention (ln_1) and the FFN (ln_2). A final LayerNorm (ln_f) follows the stack, before vocabulary projection.',
      normEvidence: 'Section 2.3 Model, page 4: normalization moves to the input of each sub-block, with an additional normalization after the last block. The original model.py names them ln_1, ln_2, and ln_f.',
      quote: 'Layer normalization […] was moved to the input of each sub-block […] and an additional layer normalization was added after the final self-attention block.',
      formula: 'x = x + MaskedSelfAttention(LayerNorm₁(x))\nx = x + FFN(LayerNorm₂(x))\nh = LayerNorm_final(h)\nlogits = h · W_tokenᵀ',
      facts: ['12 decoder blocks · 12 heads · d_model = 768 · d_ff = 3072', 'Causal self-attention; no encoder or cross-attention.', 'Learned position embeddings; GELU; vocabulary projection shares token embedding weights.'],
      note: 'The GPT-2 report has no detailed Transformer block diagram. These are original excerpts and a figure from the report. The game follows section 2.3 and the original GPT-2 source code.',
      figures: [
        figure('gpt2-pre-ln-excerpt', '§2.3 Model · Original Pre-LN excerpt', 4, 'The original text describing normalization at the sub-block input and final LayerNorm. This is a paper excerpt, not an architecture figure.'),
        figure('gpt2-table-2', 'Table 2 · GPT-2 model sizes', 4, 'Parameters, layers, and d_model for four variants. This level uses the smallest variant: 12 layers and d_model = 768.'),
        figure('gpt2-figure-2', 'Figure 2 · Children’s Book Test results', 5, 'The original plot of performance versus model size. It shows results, not the attention or FFN architecture.'),
      ],
    },
    qwen: {
      model: 'Qwen3.8-27B · text backbone', title: 'Qwen3.8-27B — official model card and configuration',
      authors: 'Qwen Team', year: '2026 · model repository created August 5',
      url: 'https://huggingface.co/Qwen/Qwen3.8-27B',
      config: 'https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json',
      code: 'https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py',
      norm: 'Pre-RMSNorm + final RMSNorm', normSummary: 'RMSNorm precedes the token mixer and SwiGLU, with a final RMSNorm after the stack. Full attention also normalizes Q/K per head; DeltaNet L2-normalizes Q/K and applies a gated output RMSNorm.',
      normEvidence: 'The official configuration identifies the Qwen3.5 architectural family, 64 layer_types, RMSNorm epsilon 1e-6, and output_gate_type = swish. Its inherited DecoderLayer applies input_layernorm and post_attention_layernorm before the mixer and MLP, with residual addition afterward; TextModel applies final norm.',
      formula: '16 × [3 × DeltaNet layer → 1 × gated-attention layer]\nx = x + Mixer(RMSNorm(x))\nx = x + SwiGLU(RMSNorm(x))\nSwiGLU(x) = down(SiLU(gate(x)) × up(x))\nh = RMSNorm_final(h)',
      facts: ['27B dense model · 64 text layers · d_model = 5120 · d_ff = 17408', '48 Gated DeltaNet layers + 16 causal gated-attention layers, in groups of four.', 'Attention: 24 Q heads, 4 K/V heads, d_head = 256; partial RoPE rotates 64 features.', 'DeltaNet: 16 Q/K heads, 48 V heads, d_head = 128; causal convolution kernel = 4.', 'No learned position vectors added to text embeddings. SwiGLU has separate gate and value branches.'],
      note: 'This level builds the text backbone. The checkpoint also contains a vision encoder; vision, video processing, and training-only multi-token prediction are outside this build. The official model card provides the layer layout instead of a paper diagram. The checkpoint specifies a swish output gate; the older inherited implementation linked here uses sigmoid, so the game follows this checkpoint’s configuration for that gate.',
      figures: [],
      documents: [{ title: 'Original checkpoint configuration', file: 'assets/qwen3.8-27b-config.json' }, { title: 'Original model-card architecture excerpt', file: 'assets/qwen3.8-27b-model-card.md' }],
    },
    encdec: {
      model: 'Attention Is All You Need', title: 'Attention Is All You Need',
      authors: 'Vaswani et al.', year: '2017',
      url: 'https://arxiv.org/abs/1706.03762', pdf: 'https://arxiv.org/pdf/1706.03762',
      norm: 'Post-LN', normSummary: 'Residual Add, then LayerNorm after each encoder and decoder sublayer. This is the Add & Norm order in Figure 1; there is no extra final LayerNorm.',
      normEvidence: 'Section 3.1 Encoder and Decoder Stacks, page 3, explicitly specifies LayerNorm(x + Sublayer(x)).',
      formula: 'x = LayerNorm(x + Sublayer(x))\nFFN(x) = ReLU(xW₁ + b₁)W₂ + b₂',
      facts: ['6 encoder blocks + 6 decoder blocks · 8 heads · d_model = 512 · d_ff = 2048', 'Encoder: bidirectional self-attention. Decoder: causal self-attention and cross-attention.', 'Sinusoidal positions; ReLU in the FFN; cross-attention K/V come from the encoder.'],
      figures: [
        figure('transformer-figure-1', 'Figure 1 · Original Transformer architecture', 3, 'Encoder on the left, decoder on the right. The flow goes upward. Add & Norm = residual Add → LayerNorm, or Post-LN.'),
        figure('transformer-figure-2', 'Figure 2 · Scaled Dot-Product and Multi-Head Attention', 4, 'The original attention operations and parallel heads. These are the operations you build manually in the game.'),
      ],
    },
  };
})();
