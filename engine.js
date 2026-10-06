(function (root) {
  'use strict';
  const BLOCKS = [
    { id: 'sliding-causal', name: 'Sliding Causal Mask', symbol: '◪', group: 'attention', desc: 'Hide future tokens and tokens outside a rolling window. Mellum2 uses a 1024-token window.' },
    { id: 'rope-yarn', name: 'RoPE + YaRN', symbol: '↻Y', group: 'input', desc: 'Apply rotary positions with YaRN scaling for long context. Mellum2 uses this in full-attention layers.' },
    { id: 'top-k', name: 'Top-k Experts', symbol: 'TopK', group: 'ffn', desc: 'Choose the experts with the largest routing probabilities, independently for each token.' },
    { id: 'renormalize', name: 'Renormalize Weights', symbol: 'Σ1', group: 'ffn', desc: 'Divide the selected routing weights by their sum, so they sum to one.' },
    { id: 'dispatch', name: 'Dispatch to Experts', symbol: '↗↘', group: 'ffn', desc: 'Send each token to its selected experts. Each expert receives the same token input.' },
    { id: 'weighted-sum', name: 'Weighted Expert Sum', symbol: 'Σw', group: 'ffn', desc: 'Multiply expert outputs by their routing weights and sum them back into one token vector.' },
    { id: 'multiply', name: 'Multiply', symbol: '×', group: 'ffn', desc: 'Multiply two branches element by element. This is the gate in SwiGLU and gated attention.' },
    { id: 'causal-conv', name: 'Causal Depthwise Conv', symbol: 'Conv', group: 'attention', desc: 'A width-4 depthwise convolution of Q/K/V. Only current and earlier tokens contribute.' },
    { id: 'l2norm', name: 'L2 Normalize Q/K', symbol: 'L2', group: 'norm', desc: 'Normalize Q and K by their vector lengths, separately for each head.' },
    { id: 'repeat-kv', name: 'Repeat Head Groups', symbol: '↟', group: 'attention', desc: 'Share K/V across query groups in GQA, or repeat DeltaNet Q/K to match its value heads.' },
    { id: 'softplus', name: 'Softplus', symbol: 'ln+', group: 'ffn', desc: 'log(1 + exp(x)). Produces the positive time step used by DeltaNet decay.' },
    { id: 'decay', name: 'Exponential Decay', symbol: 'eᵍ', group: 'attention', desc: 'alpha = exp(-exp(A_log) × softplus(a + dt_bias)). Decays the recurrent state.' },
    { id: 'delta-rule', name: 'Gated Delta Update', symbol: 'ΔS', group: 'attention', desc: 'Decay state S, then write beta × k × (v − kᵀS)ᵀ. Read the updated state with Q.' },
    { id: 'embedding', name: 'Token Embedding', symbol: 'E', group: 'input', desc: 'Map a token ID to a vector with d_model features.' },
    { id: 'learned-position', name: 'Learned Position', symbol: 'P', group: 'input', desc: 'Learned position embeddings, used in the original BERT and GPT-2.' },
    { id: 'sinusoidal', name: 'Sinusoidal Position', symbol: '∿', group: 'input', desc: 'Fixed sine and cosine position vectors from the 2017 Transformer.' },
    { id: 'segment', name: 'Token Type Embedding', symbol: 'A/B', group: 'input', desc: 'Segment A/B embeddings from the original BERT.' },
    { id: 'rope', name: 'RoPE', symbol: '↻', group: 'input', desc: 'Rotate Q/K inside attention. RoPE is not a vector added to token embeddings.' },
    { id: 'linear', name: 'Linear', symbol: 'Wx', group: 'attention', desc: 'A learned linear projection for Q/K/V, attention output, or the FFN.' },
    { id: 'split', name: 'Split Heads', symbol: '⑂', group: 'attention', desc: 'Reshape Q/K/V projections into heads. Head width may be specified independently of d_model.' },
    { id: 'qk', name: 'Q × Kᵀ', symbol: 'QKᵀ', group: 'attention', desc: 'Compare queries with keys. Scores have shape T_query × T_key.' },
    { id: 'scale', name: 'Scale 1/√dₖ', symbol: '÷√d', group: 'attention', desc: 'Scale attention scores by the size of one head, before softmax.' },
    { id: 'causal', name: 'Causal Mask', symbol: '◩', group: 'attention', desc: 'Set future-token scores to negative infinity before softmax.' },
    { id: 'bidirectional', name: 'Full Attention', symbol: '▦', group: 'attention', desc: 'Allow all valid context positions. Padding is still masked out.' },
    { id: 'softmax', name: 'Softmax', symbol: 'σ', group: 'attention', desc: 'Turn scores into probabilities: over attention keys, router experts, or vocabulary tokens. Probabilities sum to 1.' },
    { id: 'av', name: 'Attention × V', symbol: 'AV', group: 'attention', desc: 'Compute a weighted sum of values in each head.' },
    { id: 'concat', name: 'Concat Heads', symbol: '⊕', group: 'attention', desc: 'Concatenate the heads into the attention output width, then project to d_model.' },
    { id: 'add', name: 'Add', symbol: '+', group: 'norm', desc: 'Sum embeddings or add the residual: x + F(x). Tensor shapes must match.' },
    { id: 'layernorm', name: 'LayerNorm', symbol: 'LN', group: 'norm', desc: 'Normalize each token across features, with mean subtraction and learned scale/bias.' },
    { id: 'rmsnorm', name: 'RMSNorm', symbol: 'RMS', group: 'norm', desc: 'Normalize by RMS without mean subtraction. Used in models such as Llama.' },
    { id: 'batchnorm', name: 'BatchNorm', symbol: 'BN', group: 'norm', desc: 'Use batch statistics. It does not replace LayerNorm in these reference architectures.' },
    { id: 'dropout', name: 'Dropout', symbol: '∴', group: 'norm', desc: 'Randomly drop activations during training; disabled during inference.' },
    { id: 'gelu', name: 'GELU', symbol: '≈', group: 'ffn', desc: 'The smooth activation used in BERT and GPT-2; GELU variants differ slightly.' },
    { id: 'relu', name: 'ReLU', symbol: '⌞', group: 'ffn', desc: 'max(0, x). The FFN activation in the original Transformer.' },
    { id: 'silu', name: 'SiLU / Swish', symbol: '∫', group: 'ffn', desc: 'x * sigmoid(x). A SiLU layer alone does not implement the gated SwiGLU architecture.' },
    { id: 'sigmoid', name: 'Sigmoid', symbol: 'S', group: 'ffn', desc: 'Map values into (0, 1). Available for experiments.' },
    { id: 'tanh', name: 'Tanh', symbol: '~', group: 'ffn', desc: 'Map values into (-1, 1). Available for experiments.' },
  ];
  const BY_ID = Object.fromEntries(BLOCKS.map(b => [b.id, b]));
  const COARSE_BLOCKS = [
    { id: 'sliding-self-attention', name: 'Sliding Self-Attention', symbol: 'SWA', group: 'attention', desc: 'Causal self-attention restricted to a local rolling window.' },
    { id: 'mixture-of-experts', name: 'MoE Feed Forward', symbol: 'MoE', group: 'ffn', desc: 'Route each token to a subset of independent SwiGLU experts, then combine their outputs.' },
    { id: 'gated-delta-net', name: 'Gated DeltaNet', symbol: 'Δ', group: 'attention', desc: 'A causal recurrent token mixer with a decaying matrix memory and a delta-rule update.' },
    { id: 'gated-self-attention', name: 'Gated Self-Attention', symbol: 'GQA', group: 'attention', desc: 'Causal grouped-query attention with Q/K normalization, partial RoPE, and an output gate.' },
    { id: 'input-embeddings', name: 'Input Embeddings', symbol: 'E', group: 'input', desc: 'Turn token IDs and, when used, position embeddings into the stack input. Build the embedding operations inside later.' },
    { id: 'self-attention', name: 'Self-Attention', symbol: 'MHA', group: 'attention', desc: 'Multi-head self-attention with access to the full valid context.' },
    { id: 'masked-self-attention', name: 'Masked Self-Attention', symbol: '◩', group: 'attention', desc: 'Multi-head self-attention that hides future tokens.' },
    { id: 'cross-attention', name: 'Cross-Attention', symbol: 'Q/KV', group: 'attention', desc: 'Queries come from the decoder; keys and values come from encoder memory.' },
    { id: 'feed-forward', name: 'Feed Forward', symbol: 'FFN', group: 'ffn', desc: 'A token-wise network. Build its projections and activations inside later.' },
    { id: 'add-norm', name: 'Add & Norm', symbol: '+LN', group: 'norm', desc: 'Post-LN: add the original x to the sublayer output, then normalize.' },
    { id: 'normalization', name: 'Normalization', symbol: 'LN', group: 'norm', desc: 'A separate normalization layer. Build its normalization operation inside.' },
    { id: 'residual-add', name: 'Residual Add', symbol: '+', group: 'norm', desc: 'Add the original sublayer input to its output. Build the Add operation inside.' },
    { id: 'lm-head', name: 'Language Model Head', symbol: 'LM', group: 'output', desc: 'Project decoder states to the vocabulary and apply softmax. A Pre-LN stack also needs final normalization here.' },
    { id: 'final-norm', name: 'Final Normalization', symbol: 'LNf', group: 'norm', desc: 'Normalize the final encoder output when experimenting with a Pre-LN encoder stack.' },
    { id: 'pooling', name: 'Global Pooling', symbol: '↓T', group: 'output', desc: 'Merge token states into one vector. These reference models keep one state per token.' },
    { id: 'recurrent', name: 'Recurrent Layer', symbol: 'RNN', group: 'attention', desc: 'Process a sequence recurrently. The reference Transformer blocks use attention and token-wise feed-forward networks.' },
  ];
  const COARSE_BY_ID = Object.fromEntries(COARSE_BLOCKS.map(b => [b.id, b]));
  COARSE_BY_ID['pre-residual'] = { id: 'pre-residual', name: 'Pre-LN Residual', symbol: 'LN+', group: 'norm', desc: 'Normalize before the sublayer and add the original input afterward. The residual bypasses normalization.' };
  const PROFILES = {
    bert: { name: 'BERT', year: 2018, eyebrow: '01 / ENCODER', subtitle: 'Build a bidirectional encoder.', desc: 'Build the original BERT encoder. Every token can attend to the full valid context.', label: 'BERT-base', d: 768, heads: 12, depth: 12, hidden: 3072, norm: 'post', activation: 'gelu', position: 'learned-position', tokens: ['[CLS]', 'cat', 'sits', 'on', 'mat', '[SEP]'] },
    gpt: { name: 'GPT-2', year: 2019, eyebrow: '02 / DECODER', subtitle: 'Build a decoder that predicts the next token.', desc: 'Build the original GPT-2 decoder: causal attention, Pre-LN, and a final LayerNorm.', label: 'GPT-2 small', d: 768, heads: 12, depth: 12, hidden: 3072, norm: 'pre', activation: 'gelu', position: 'learned-position', tokens: ['cat', 'sits', 'on', 'mat', 'and', '…'] },
    encdec: { name: 'Attention Is All You Need', year: 2017, eyebrow: '03 / SEQUENCE TO SEQUENCE', subtitle: 'Build the original encoder and decoder.', desc: 'Build the Transformer from Attention Is All You Need. The decoder attends to the encoder output.', label: 'Attention Is All You Need', d: 512, heads: 8, depth: 6, hidden: 2048, norm: 'post', activation: 'relu', position: 'sinusoidal', tokens: ['<BOS>', 'the', 'cat', 'is', 'sitting', '…'] },
    qwen: { name: 'Qwen3.8-27B', year: 2026, eyebrow: '04 / HYBRID DECODER', subtitle: 'Build a modern hybrid decoder.', desc: 'Build the text backbone: three Gated DeltaNet layers, then one gated-attention layer, repeated 16 times.', label: 'Qwen3.8-27B · text backbone', d: 5120, heads: 24, kvHeads: 4, headDim: 256, depth: 64, hidden: 17408, norm: 'pre', normType: 'rmsnorm', activation: 'silu', gatedFFN: true, outputGate: true, hybridGroup: { repeats: 16, first: 3 }, position: null, tokens: ['the', 'cat', 'sits', 'on', 'mat', '…'] },
    mellum: { name: 'Mellum 2', year: 2026, eyebrow: '05 / SPARSE DECODER', subtitle: 'Build a coding model with sparse experts.', desc: 'Build Mellum2-12B-A2.5B-Instruct: three sliding-attention layers, then one full causal layer, repeated 7 times.', label: 'Mellum2-12B-A2.5B-Instruct', d: 2304, heads: 32, kvHeads: 4, headDim: 128, depth: 28, hidden: 896, denseHidden: 7168, experts: 64, topK: 8, window: 1024, norm: 'pre', normType: 'rmsnorm', activation: 'silu', gatedFFN: true, hybridGroup: { repeats: 7, first: 3 }, position: null, tokens: ['def', 'add', '(', 'a', ',', 'b'] },
  };
  const ATTENTION_STEPS = [
    { key: 'q', label: 'Project queries (Q)', expected: 'linear' },
    { key: 'k', label: 'Project keys (K)', expected: 'linear' },
    { key: 'v', label: 'Project values (V)', expected: 'linear' },
    { key: 'split', label: 'Split into heads', expected: 'split' },
    { key: 'scores', label: 'Compare Q and K', expected: 'qk' },
    { key: 'scale', label: 'Scale scores', expected: 'scale' },
    { key: 'mask', label: 'Choose an attention mask', expected: null },
    { key: 'softmax', label: 'Normalize attention weights', expected: 'softmax' },
    { key: 'weighted', label: 'Weight the values (V)', expected: 'av' },
    { key: 'concat', label: 'Merge the heads', expected: 'concat' },
    { key: 'out', label: 'Project attention output', expected: 'linear' },
  ];
  const step = (key, label, expected) => ({ key, label, expected });
  const DELTA_STEPS = [
    step('qkv', 'Project Q, K, V', 'linear'), step('conv', 'Mix nearby tokens', 'causal-conv'),
    step('convAct', 'Activate convolved Q/K/V', 'silu'), step('split', 'Split into heads', 'split'),
    step('qkNorm', 'Normalize Q/K vectors', 'l2norm'), step('repeat', 'Match Q/K and V heads', 'repeat-kv'),
    step('scale', 'Scale queries', 'scale'), step('betaProj', 'Project update strength', 'linear'),
    step('beta', 'Bound update strength', 'sigmoid'), step('decayProj', 'Project decay time step', 'linear'),
    step('timeStep', 'Make time step positive', 'softplus'), step('decay', 'Decay the recurrent state', 'decay'),
    step('rule', 'Update and read memory', 'delta-rule'), step('outNorm', 'Normalize each output head', 'rmsnorm'),
    step('gateProj', 'Project output gate', 'linear'), step('gateAct', 'Activate output gate', 'silu'),
    step('gate', 'Apply output gate', 'multiply'), step('concat', 'Merge value heads', 'concat'),
    step('out', 'Project back to model width', 'linear'),
  ];
  const SWIGLU_STEPS = [step('gateProj', 'Project gate branch', 'linear'), step('upProj', 'Project value branch', 'linear'), step('activation', 'Activate gate branch', 'silu'), step('multiply', 'Multiply gate and value branches', 'multiply'), step('downProj', 'Project back to model width', 'linear')];
  const MOE_STEPS = [step('routerProj', 'Project router logits', 'linear'), step('routerSoftmax', 'Normalize over all experts', 'softmax'), step('topK', 'Select experts per token', 'top-k'), step('renormalize', 'Normalize selected routing weights', 'renormalize'), step('dispatch', 'Send tokens to selected experts', 'dispatch'), ...SWIGLU_STEPS, step('combine', 'Combine weighted expert outputs', 'weighted-sum')];
  function headDim(game) { return PROFILES[game.level].headDim || game.d / game.heads; }
  function moduleSteps(game, meta) {
    if (meta.kind === 'delta') return DELTA_STEPS;
    if (meta.kind === 'moe') return MOE_STEPS;
    if (meta.kind === 'ffn') return PROFILES[game.level].gatedFFN ? SWIGLU_STEPS : null;
    const p = PROFILES[game.level];
    if (!p.kvHeads) return ATTENTION_STEPS.map(s => ({ ...s, expected: s.key === 'mask' ? meta.mask : s.expected }));
    return [
      ...ATTENTION_STEPS.slice(0, 4), step('qNorm', 'Normalize query heads', 'rmsnorm'), step('kNorm', 'Normalize key heads', 'rmsnorm'),
      step('rope', 'Rotate Q/K positions', meta.rope || 'rope'), step('repeat', 'Share K/V across query groups', 'repeat-kv'),
      ...ATTENTION_STEPS.slice(4, 10).map(s => ({ ...s, expected: s.key === 'mask' ? meta.mask : s.expected })),
      ...(p.outputGate ? [step('gateProj', 'Project output gate', 'linear'), step('gateAct', 'Activate output gate', 'silu'), step('gate', 'Apply output gate', 'multiply')] : []), ATTENTION_STEPS[10],
    ];
  }
  const clone = value => JSON.parse(JSON.stringify(value));
  function createGame(level) {
    const p = PROFILES[level];
    return { level, norm: p.norm, experimental: false, d: p.d, heads: p.heads, hidden: p.hidden, depth: p.depth, structure: {}, slots: {}, modules: {}, completed: false, attempts: 0, hints: 0 };
  }
  function lanes(game) {
    return game.level === 'encdec' ? ['encoder', 'decoder'] : [game.level === 'bert' ? 'encoder' : 'decoder'];
  }
  function modulesFor(game, lane) {
    if (game.level === 'mellum') return [
      { id: 'decoder.local', kind: 'attention', name: 'Sliding Self-Attention', mask: 'sliding-causal', rope: 'rope' },
      { id: 'decoder.local.ffn', kind: 'moe', name: 'MoE Feed Forward' },
      { id: 'decoder.self', kind: 'attention', name: 'Causal Self-Attention', mask: 'causal', rope: 'rope-yarn' },
      { id: 'decoder.ffn', kind: 'moe', name: 'MoE Feed Forward' },
    ];
    if (game.level === 'qwen') return [
      { id: 'decoder.delta', kind: 'delta', name: 'Gated DeltaNet', mask: 'causal' },
      { id: 'decoder.delta.ffn', kind: 'ffn', name: 'Feed Forward · SwiGLU' },
      { id: 'decoder.self', kind: 'attention', name: 'Gated Self-Attention', mask: 'causal' },
      { id: 'decoder.ffn', kind: 'ffn', name: 'Feed Forward · SwiGLU' },
    ];
    return [{ id: lane + '.self', kind: 'attention', name: lane === 'decoder' ? 'Masked Self-Attention' : 'Self-Attention', mask: lane === 'decoder' ? 'causal' : 'bidirectional' }, ...(game.level === 'encdec' && lane === 'decoder' ? [{ id: lane + '.cross', kind: 'attention', name: 'Cross-Attention', mask: 'bidirectional' }] : []), { id: lane + '.ffn', kind: 'ffn', name: 'Feed Forward' }];
  }
  function coarseTopology(game) {
    const stages = [];
    for (const lane of lanes(game)) {
      const push = (id, scope, expected) => stages.push({ id, lane, scope, label: `${lane === 'encoder' ? 'Encoder' : 'Decoder'} block ${stages.filter(s => s.lane === lane).length + 1}`, expected });
      push(lane + '.input', 'embedding', 'input-embeddings');
      for (const meta of modulesFor(game, lane)) {
        if (game.norm === 'pre') push(meta.id + '.norm', meta.id, 'normalization');
        push(meta.id, meta.id, meta.kind === 'moe' ? 'mixture-of-experts' : meta.mask === 'sliding-causal' ? 'sliding-self-attention' : meta.kind === 'ffn' ? 'feed-forward' : meta.kind === 'delta' ? 'gated-delta-net' : game.level === 'qwen' ? 'gated-self-attention' : meta.id.endsWith('.cross') ? 'cross-attention' : lane === 'decoder' ? 'masked-self-attention' : 'self-attention');
        push(meta.id + (game.norm === 'pre' ? '.add' : '.residual'), meta.id, game.norm === 'pre' ? 'residual-add' : 'add-norm');
      }
      if (lane === 'decoder' || game.norm === 'pre') push(lane + '.output', 'output', lane === 'decoder' ? 'lm-head' : 'final-norm');
    }
    return stages;
  }
  function ensureStructure(game) {
    if (game.structure && typeof game.structure === 'object' && !Array.isArray(game.structure)) {
      // Earlier Pre-LN cards combined operations on opposite sides of a sublayer.
      // Split their outer placement while keeping every manually built operation.
      for (const [id, value] of Object.entries(game.structure)) if (id.endsWith('.residual') && value === 'pre-residual') {
        const module = id.slice(0, -'.residual'.length);
        game.structure[module + '.norm'] ||= 'normalization';
        game.structure[module + '.add'] ||= 'residual-add';
        delete game.structure[id];
      }
      return game.structure;
    }
    // Old saves already displayed the outer modules. Preserve all work inside them.
    game.structure = {};
    for (const stage of coarseTopology(game)) {
      let touched = game.completed;
      if (stage.scope === 'embedding' || stage.scope === 'output') touched ||= topology(game).some(s => s.lane === stage.lane && s.scope === stage.scope && game.slots[s.id]);
      else {
        const m = game.modules[stage.scope];
        touched ||= m?.kind === 'ffn' && m.layers ? m.layers.some(l => l.type) : Object.values(m?.slots || {}).some(Boolean);
        touched ||= game.slots[stage.scope + '.norm'] || game.slots[stage.scope + '.add'];
      }
      if (touched) game.structure[stage.id] = stage.expected;
    }
    return game.structure;
  }
  function checkStructure(game) {
    const structure = ensureStructure(game), p = PROFILES[game.level], checks = [];
    const record = (id, label, ok, message) => checks.push({ id, label, ok, message });
    record('config.norm', 'Reference normalization order', game.experimental || game.norm === p.norm, `${p.label} uses ${p.norm === 'pre' ? (p.normType === 'rmsnorm' ? 'Pre-RMSNorm with a final RMSNorm' : 'Pre-LN with a final LayerNorm') : 'Post-LN'}. Restore that order or enable Experiment mode to try a different architecture.`);
    record('config.heads', 'Head size', p.kvHeads ? Number.isInteger(game.heads) && game.heads > 0 && game.heads % p.kvHeads === 0 : Number.isInteger(game.d / game.heads) && game.heads > 0, p.kvHeads ? `Query heads must be divisible by the ${p.kvHeads} K/V heads. Head width is explicitly ${p.headDim}, independent of d_model.` : `d_model (${game.d}) must be divisible by h (${game.heads}) with no remainder.`);
    if (p.kvHeads && !game.experimental) for (const key of ['d', 'heads', 'hidden', 'depth']) record('config.' + key, 'Reference ' + key, game[key] === p[key], `${p.label} uses ${key} = ${p[key]}. Enable Experiment mode to change dimensions.`);
    for (const stage of coarseTopology(game)) {
      const actual = structure[stage.id], expected = COARSE_BY_ID[stage.expected];
      const message = !actual ? `Place a module in "${stage.label}".` : `${COARSE_BY_ID[actual]?.name || actual} does not fit "${stage.label}". Use ${expected.name}.`;
      record('structure.' + stage.id, stage.label, actual === stage.expected, message);
    }
    const issues = checks.filter(c => !c.ok);
    return { kind: 'structure', valid: !issues.length, checks, issues, notes: [], correct: checks.length - issues.length, total: checks.length };
  }
  function coarseProgress(game) {
    const structure = ensureStructure(game), stages = coarseTopology(game);
    return { filled: stages.filter(s => structure[s.id]).length, total: stages.length };
  }
  function topology(game) {
    const p = PROFILES[game.level];
    const result = [];
    const push = (id, label, expected, lane, scope = 'outside') => result.push({ id, label, expected, lane, scope });
    for (const lane of lanes(game)) {
      push(lane + '.token', 'Token vectors', 'embedding', lane, 'embedding');
      if (p.position) push(lane + '.position', 'Position vectors', p.position, lane, 'embedding');
      if (game.level === 'bert') push(lane + '.segment', 'Segment A/B vectors', 'segment', lane, 'embedding');
      if (p.position) push(lane + '.embedAdd', 'Combine the input vectors', 'add', lane, 'embedding');
      if (game.level === 'bert') push(lane + '.embedNorm', 'Normalize input embeddings', 'layernorm', lane, 'embedding');
      for (const m of modulesFor(game, lane)) {
        push(m.id + '.norm', game.norm === 'pre' ? 'Normalize before ' + (m.kind === 'moe' ? 'MoE' : m.kind === 'ffn' ? 'FFN' : 'attention') : 'Normalize after residual addition', p.normType || 'layernorm', lane, m.id);
        push(m.id + '.add', 'Residual: x + F(x)', 'add', lane, m.id);
      }
      if (game.norm === 'pre') push(lane + '.finalNorm', 'Normalize the final stack output', p.normType || 'layernorm', lane, 'output');
      if (lane === 'decoder') {
        push(lane + '.head', 'Project to the vocabulary', 'linear', lane, 'output');
        push(lane + '.probs', 'Next-token probabilities', 'softmax', lane, 'output');
      }
    }
    return result;
  }
  function ensureModule(game, id) {
    if (!game.modules[id]) {
      if (PROFILES[game.level].gatedFFN && id.endsWith('.ffn')) {
        const p = PROFILES[game.level];
        game.modules[id] = { kind: p.experts ? 'moe' : 'ffn', gated: true, slots: {}, widths: { gateProj: game.hidden, upProj: game.hidden, downProj: game.d }, ...(p.experts ? { routing: { experts: p.experts, topK: p.topK } } : {}) };
        return game.modules[id];
      }
      game.modules[id] = id === 'decoder.delta' ? { kind: 'delta', slots: {} } : id.endsWith('.ffn') ? { kind: 'ffn', layers: [{ type: null, width: game.hidden }, { type: null, width: game.hidden }, { type: null, width: game.d }] } : { kind: 'attention', slots: {}, sources: { q: 'hidden', k: 'hidden', v: 'hidden' } };
    }
    return game.modules[id];
  }
  function check(game) {
    const outer = checkStructure(game);
    const issues = [...outer.issues], checks = [...outer.checks], notes = [];
    const p = PROFILES[game.level];
    const record = (id, label, ok, message, module) => { checks.push({ id, label, ok, message, module }); if (!ok) issues.push({ id, label, message, module }); };
    for (const s of topology(game)) {
      const actual = game.slots[s.id];
      const normAlt = game.experimental && ((s.expected === 'layernorm' && actual === 'rmsnorm') || (s.expected === 'rmsnorm' && actual === 'layernorm'));
      const ok = actual === s.expected || normAlt;
      let message = !actual ? `Fill the "${s.label}" slot.` : `${BY_ID[actual]?.name || actual} does not fit here. Use ${BY_ID[s.expected].name}.`;
      if (actual === 'rope') message = 'RoPE rotates Q/K inside attention. This slot needs position vectors added to token embeddings.';
      if (actual === 'batchnorm') message = `BatchNorm uses batch statistics. This slot needs per-token feature normalization: ${BY_ID[s.expected].name}.`;
      record(s.id, s.label, ok, message);
      if (normAlt) notes.push(`${BY_ID[actual].name} is allowed in Experiment mode. The original architecture uses ${BY_ID[s.expected].name}.`);
    }
    for (const lane of lanes(game)) for (const meta of modulesFor(game, lane)) {
      const m = ensureModule(game, meta.id);
      if (meta.kind !== 'ffn' || m.gated) {
        for (const step of moduleSteps(game, meta)) {
          const expected = step.expected;
          const actual = m.slots[step.key];
          let message = !actual ? `Add an operation to ${step.label.toLowerCase()}.` : `For "${step.label}", use ${BY_ID[expected].name}.`;
          if (step.key === 'mask' && actual !== expected) message = meta.mask === 'sliding-causal' ? `Use a sliding causal mask with a ${p.window}-token window. A full causal mask can see older tokens outside this window.` : meta.mask === 'causal' ? 'Decoder self-attention must not see future tokens. Apply a causal mask before softmax.' : meta.id.endsWith('.cross') ? 'Cross-attention can see the full encoder input. Decoder self-attention masks future output tokens.' : 'The encoder needs context on both sides. A causal mask would make it autoregressive.';
          record(meta.id + '.' + step.key, step.label, actual === expected, message, meta.id);
        }
        if (meta.kind === 'attention') for (const role of ['q', 'k', 'v']) {
          const expectedSource = meta.id.endsWith('.cross') && role !== 'q' ? 'encoder' : 'hidden';
          record(meta.id + '.source.' + role, `${role.toUpperCase()} source`, m.sources[role] === expectedSource, expectedSource === 'encoder' ? `${role.toUpperCase()} must come from the encoder output. Only Q comes from the decoder.` : `${role.toUpperCase()} must come from the current ${lane} hidden states.`, meta.id);
        }
        if (meta.kind === 'moe') {
          record(meta.id + '.experts', 'Expert count', m.routing?.experts === p.experts, `Mellum2 has ${p.experts} independent experts per layer.`, meta.id);
          record(meta.id + '.routing.topK', 'Active experts per token', m.routing?.topK === p.topK, `Mellum2 selects ${p.topK} of ${p.experts} experts per token.`, meta.id);
        }
        if (m.gated) {
          for (const key of ['gateProj', 'upProj', 'downProj']) {
            const expectedWidth = key === 'downProj' ? game.d : game.hidden;
            record(meta.id + '.width.' + key, 'SwiGLU branch width', m.widths[key] === expectedWidth, `${key} must output ${expectedWidth} features so the gated branches and residual match.`, meta.id);
          }
        }
      } else {
        const layers = m.layers;
        if (!game.experimental) {
          record(meta.id + '.length', 'FFN structure', layers.length === 3, 'The reference FFN is Linear → activation → Linear. Enable Experiment mode for more layers.', meta.id);
          const expected = ['linear', p.activation, 'linear'];
          for (let i = 0; i < Math.max(layers.length, 3); i++) {
            const actual = layers[i]?.type;
            record(meta.id + '.layer.' + i, 'FFN layer ' + (i + 1), actual === expected[i] && !!expected[i], !actual ? `Add ${BY_ID[expected[i]]?.name || 'a layer'} at FFN position ${i + 1}.` : `Reference position ${i + 1} needs ${BY_ID[expected[i]]?.name || 'the end of the FFN'}.`, meta.id);
          }
          if (layers[0]?.type === 'linear') record(meta.id + '.width.0', 'FFN expansion', layers[0].width === game.hidden, `The first projection must expand d_model to d_ff = ${game.hidden}.`, meta.id);
        } else {
          record(meta.id + '.length', 'FFN has enough operations', layers.length >= 3, 'Add at least Linear → activation → Linear.', meta.id);
          for (let i = 0; i < layers.length; i++) {
            const type = layers[i].type;
            const supported = ['linear', 'gelu', 'relu', 'silu', 'sigmoid', 'tanh', 'dropout', 'layernorm', 'rmsnorm'].includes(type);
            record(meta.id + '.layer.' + i, 'FFN layer ' + (i + 1), supported, type ? `${BY_ID[type]?.name || type} is not a supported token-wise FFN operation.` : 'Choose an operation for this FFN layer.', meta.id);
          }
          record(meta.id + '.start', 'FFN input projection', layers[0]?.type === 'linear', 'Start the FFN with Linear.', meta.id);
          record(meta.id + '.end', 'FFN output projection', layers.at(-1)?.type === 'linear', 'End the FFN with Linear.', meta.id);
          record(meta.id + '.nonlinearity', 'FFN nonlinearity', layers.some(l => ['gelu', 'relu', 'silu', 'sigmoid', 'tanh'].includes(l.type)), 'Add a nonlinear activation. A chain of Linear layers collapses into one linear operation.', meta.id);
        }
        let width = game.d;
        for (let i = 0; i < layers.length; i++) if (layers[i].type === 'linear') {
          record(meta.id + '.shape.' + i, 'Layer width ' + (i + 1), Number.isInteger(Number(layers[i].width)) && Number(layers[i].width) > 0, 'A Linear output width must be a positive integer.', meta.id);
          width = Number(layers[i].width);
        }
        record(meta.id + '.residual', 'Residual shape compatibility', width === game.d, `The FFN outputs ${width} features; the residual has ${game.d}. The final Linear must return d_model = ${game.d}.`, meta.id);
        if (game.experimental && (layers.length !== 3 || layers[1]?.type !== p.activation)) notes.push('This FFN differs from the reference. Its tensor widths have been checked.');
      }
    }
    if (game.norm !== p.norm) notes.push(`${p.label} uses ${p.norm === 'pre' ? 'Pre-LN' : 'Post-LN'}. This is an experimental variant with ${game.norm === 'pre' ? 'Pre-LN' : 'Post-LN'}.`);
    return { valid: !issues.length, issues, checks, notes: [...new Set(notes)], correct: checks.filter(c => c.ok).length, total: checks.length };
  }
  function progress(game) {
    const outer = coarseProgress(game);
    let filled = outer.filled, total = outer.total + topology(game).length;
    for (const s of topology(game)) if (game.slots[s.id]) filled++;
    for (const lane of lanes(game)) for (const meta of modulesFor(game, lane)) {
      const m = ensureModule(game, meta.id);
      const steps = moduleSteps(game, meta);
      if (steps) { total += steps.length; filled += steps.filter(s => m.slots[s.key]).length; }
      else { total += m.layers.length; filled += m.layers.filter(l => l.type).length; }
    }
    return { filled, total, percent: total ? Math.round(filled / total * 100) : 0 };
  }
  function fillReference(game) {
    const structure = ensureStructure(game);
    for (const s of coarseTopology(game)) structure[s.id] = s.expected;
    for (const s of topology(game)) game.slots[s.id] = s.expected;
    for (const lane of lanes(game)) for (const meta of modulesFor(game, lane)) {
      const m = ensureModule(game, meta.id);
      const steps = moduleSteps(game, meta);
      if (steps) {
        for (const s of steps) m.slots[s.key] = s.expected;
        if (meta.kind === 'moe') m.routing = { experts: PROFILES[game.level].experts, topK: PROFILES[game.level].topK };
        if (m.gated) m.widths = { gateProj: game.hidden, upProj: game.hidden, downProj: game.d };
        if (meta.kind === 'attention') for (const role of ['q', 'k', 'v']) m.sources[role] = meta.id.endsWith('.cross') && role !== 'q' ? 'encoder' : 'hidden';
      } else m.layers = [{ type: 'linear', width: game.hidden }, { type: PROFILES[game.level].activation, width: game.hidden }, { type: 'linear', width: game.d }];
    }
    return game;
  }
  function trace(game) {
    const steps = [];
    for (const lane of lanes(game)) {
      const t = topology(game).filter(s => s.lane === lane);
      for (const s of t.filter(s => s.scope === 'embedding')) steps.push({ id: s.id, lane, label: BY_ID[game.slots[s.id]]?.name || s.label, desc: `Token tensor: [batch, T_${lane}, ${game.d}]` });
      for (const meta of modulesFor(game, lane)) {
        const shape = `[batch, T_${lane}, ${game.d}]`;
        if (game.norm === 'pre') steps.push({ id: meta.id + '.norm', lane, label: 'Normalize before ' + meta.name, desc: `Normalize the sublayer input. The residual keeps the original x. ${shape}` });
        steps.push({ id: meta.id, lane, label: meta.name, desc: meta.id.endsWith('.cross') ? `Q ← decoder, K/V ← encoder. Scores: [batch, ${game.heads}, T_decoder, T_encoder].` : meta.kind === 'delta' ? '16 Q/K heads and 48 V heads, each 128 wide. Causal width-4 convolution → delta memory → gated RMSNorm → 6144 to model width.' : game.level === 'qwen' && meta.kind === 'attention' ? `${game.heads} Q heads and 4 K/V heads, each 256 wide. Q/K RMSNorm → partial RoPE (64 features) → causal attention → SiLU output gate → ${game.heads * 256} to model width.` : meta.kind === 'moe' ? `Route each token to ${PROFILES[game.level].topK} of ${PROFILES[game.level].experts} experts. Each expert: ${game.d} → ${game.hidden} → ${game.d}. Weighted outputs sum to ${shape}.` : game.level === 'mellum' && meta.kind === 'attention' ? `${game.heads} Q heads, 4 K/V heads, width 128. Q/K RMSNorm → ${meta.rope === 'rope-yarn' ? 'YaRN-scaled RoPE; all past tokens' : 'RoPE; causal window 1024'} → attention → ${game.heads * 128} to model width.` : meta.kind === 'ffn' ? `Apply the same FFN separately to every token. Output: ${shape}.` : `${game.heads} query heads × ${headDim(game)} features. ${meta.mask === 'causal' ? 'Future tokens are masked.' : 'All non-padding context is available.'}` });
        steps.push({ id: meta.id + '.add', lane, label: 'Residual Add', desc: `Add x and the sublayer output with matching shape ${shape}.` });
        if (game.norm === 'post') steps.push({ id: meta.id + '.norm', lane, label: 'Normalize after the residual', desc: `${BY_ID[PROFILES[game.level].normType || 'layernorm'].name}(x + F(x)), shape ${shape}.` });
      }
      for (const s of t.filter(s => s.scope === 'output')) steps.push({ id: s.id, lane, label: BY_ID[game.slots[s.id]]?.name || s.label, desc: s.id.endsWith('.probs') ? 'Softmax over the vocabulary gives next-token probabilities. This is separate from attention softmax.' : s.id.endsWith('.head') ? 'Linear: d_model → vocabulary_size.' : `Final normalization of the Pre-LN stack output: [batch, T_${lane}, ${game.d}].` });
    }
    return steps;
  }
  const api = { moduleSteps, headDim, MOE_STEPS, DELTA_STEPS, SWIGLU_STEPS, BLOCKS, BY_ID, COARSE_BLOCKS, COARSE_BY_ID, PROFILES, ATTENTION_STEPS, createGame, lanes, modulesFor, coarseTopology, ensureStructure, checkStructure, coarseProgress, topology, ensureModule, check, progress, fillReference, trace, clone };
  root.TransformerEngine = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
