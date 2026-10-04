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
