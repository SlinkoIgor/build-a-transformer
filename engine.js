(function (root) {
  'use strict';
  const BLOCKS = [
    { id: 'embedding', name: 'Token Embedding', symbol: 'E', group: 'input', desc: 'Map a token ID to a vector with d_model features.' },
    { id: 'learned-position', name: 'Learned Position', symbol: 'P', group: 'input', desc: 'Learned position embeddings, used in the original BERT and GPT-2.' },
    { id: 'sinusoidal', name: 'Sinusoidal Position', symbol: '∿', group: 'input', desc: 'Fixed sine and cosine position vectors from the 2017 Transformer.' },
    { id: 'segment', name: 'Token Type Embedding', symbol: 'A/B', group: 'input', desc: 'Segment A/B embeddings from the original BERT.' },
    { id: 'rope', name: 'RoPE', symbol: '↻', group: 'input', desc: 'Rotate Q/K inside attention. RoPE is not a vector added to token embeddings.' },
    { id: 'linear', name: 'Linear', symbol: 'Wx', group: 'attention', desc: 'A learned linear projection for Q/K/V, attention output, or the FFN.' },
    { id: 'split', name: 'Split Heads', symbol: '⑂', group: 'attention', desc: 'Reshape Q/K/V into h heads: d_head = d_model / h.' },
    { id: 'qk', name: 'Q × Kᵀ', symbol: 'QKᵀ', group: 'attention', desc: 'Compare queries with keys. Scores have shape T_query × T_key.' },
    { id: 'scale', name: 'Scale 1/√dₖ', symbol: '÷√d', group: 'attention', desc: 'Scale attention scores by the size of one head, before softmax.' },
    { id: 'causal', name: 'Causal Mask', symbol: '◩', group: 'attention', desc: 'Set future-token scores to negative infinity before softmax.' },
    { id: 'bidirectional', name: 'Full Attention', symbol: '▦', group: 'attention', desc: 'Allow all valid context positions. Padding is still masked out.' },
    { id: 'softmax', name: 'Softmax', symbol: 'σ', group: 'attention', desc: 'Turn scores into attention weights. Weights sum to 1 over keys.' },
    { id: 'av', name: 'Attention × V', symbol: 'AV', group: 'attention', desc: 'Compute a weighted sum of values in each head.' },
    { id: 'concat', name: 'Concat Heads', symbol: '⊕', group: 'attention', desc: 'Concatenate h heads back into d_model features.' },
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
    { id: 'input-embeddings', name: 'Input Embeddings', symbol: 'E+P', group: 'input', desc: 'Turn token IDs and positions into the input to the stack. Build the embedding operations inside later.' },
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
  const clone = value => JSON.parse(JSON.stringify(value));
  function createGame(level) {
    const p = PROFILES[level];
    return { level, norm: p.norm, experimental: false, d: p.d, heads: p.heads, hidden: p.hidden, depth: p.depth, structure: {}, slots: {}, modules: {}, completed: false, attempts: 0, hints: 0 };
  }
  function lanes(game) {
    return game.level === 'encdec' ? ['encoder', 'decoder'] : [game.level === 'bert' ? 'encoder' : 'decoder'];
  }
  function modulesFor(game, lane) {
    return [{ id: lane + '.self', kind: 'attention', name: lane === 'decoder' ? 'Masked Self-Attention' : 'Self-Attention', mask: lane === 'decoder' ? 'causal' : 'bidirectional' }, ...(game.level === 'encdec' && lane === 'decoder' ? [{ id: lane + '.cross', kind: 'attention', name: 'Cross-Attention', mask: 'bidirectional' }] : []), { id: lane + '.ffn', kind: 'ffn', name: 'Feed Forward' }];
  }
  function coarseTopology(game) {
    const stages = [];
    for (const lane of lanes(game)) {
      const push = (id, scope, expected) => stages.push({ id, lane, scope, label: `${lane === 'encoder' ? 'Encoder' : 'Decoder'} block ${stages.filter(s => s.lane === lane).length + 1}`, expected });
      push(lane + '.input', 'embedding', 'input-embeddings');
      for (const meta of modulesFor(game, lane)) {
        if (game.norm === 'pre') push(meta.id + '.norm', meta.id, 'normalization');
        push(meta.id, meta.id, meta.kind === 'ffn' ? 'feed-forward' : meta.id.endsWith('.cross') ? 'cross-attention' : lane === 'decoder' ? 'masked-self-attention' : 'self-attention');
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
        touched ||= m?.kind === 'ffn' ? m.layers.some(l => l.type) : Object.values(m?.slots || {}).some(Boolean);
        touched ||= game.slots[stage.scope + '.norm'] || game.slots[stage.scope + '.add'];
      }
      if (touched) game.structure[stage.id] = stage.expected;
    }
    return game.structure;
  }
  function checkStructure(game) {
    const structure = ensureStructure(game), p = PROFILES[game.level], checks = [];
    const record = (id, label, ok, message) => checks.push({ id, label, ok, message });
    record('config.norm', 'Reference normalization order', game.experimental || game.norm === p.norm, `${p.label} uses ${p.norm === 'pre' ? 'Pre-LN with a final LayerNorm' : 'Post-LN'}. Restore that order or enable Experiment mode to try a different architecture.`);
    record('config.heads', 'Head size', Number.isInteger(game.d / game.heads) && game.heads > 0, `d_model (${game.d}) must be divisible by h (${game.heads}) with no remainder.`);
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
      push(lane + '.position', 'Position vectors', p.position, lane, 'embedding');
      if (game.level === 'bert') push(lane + '.segment', 'Segment A/B vectors', 'segment', lane, 'embedding');
      push(lane + '.embedAdd', 'Combine the input vectors', 'add', lane, 'embedding');
      if (game.level === 'bert') push(lane + '.embedNorm', 'Normalize input embeddings', 'layernorm', lane, 'embedding');
      for (const m of modulesFor(game, lane)) {
        push(m.id + '.norm', game.norm === 'pre' ? 'Normalize before ' + (m.kind === 'ffn' ? 'FFN' : 'attention') : 'Normalize after residual addition', 'layernorm', lane, m.id);
        push(m.id + '.add', 'Residual: x + F(x)', 'add', lane, m.id);
      }
      if (game.norm === 'pre') push(lane + '.finalNorm', 'Normalize the final stack output', 'layernorm', lane, 'output');
      if (lane === 'decoder') {
        push(lane + '.head', 'Project to the vocabulary', 'linear', lane, 'output');
        push(lane + '.probs', 'Next-token probabilities', 'softmax', lane, 'output');
      }
    }
    return result;
  }
  function ensureModule(game, id) {
    if (!game.modules[id]) {
      game.modules[id] = id.endsWith('.ffn') ? { kind: 'ffn', layers: [{ type: null, width: game.hidden }, { type: null, width: game.hidden }, { type: null, width: game.d }] } : { kind: 'attention', slots: {}, sources: { q: 'hidden', k: 'hidden', v: 'hidden' } };
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
      const normAlt = game.experimental && s.expected === 'layernorm' && actual === 'rmsnorm';
      const ok = actual === s.expected || normAlt;
      let message = !actual ? `Fill the "${s.label}" slot.` : `${BY_ID[actual]?.name || actual} does not fit here. Use ${BY_ID[s.expected].name}.`;
      if (actual === 'rope') message = 'RoPE rotates Q/K inside attention. This slot needs position vectors added to token embeddings.';
      if (actual === 'batchnorm') message = 'BatchNorm uses batch statistics. This slot needs per-token feature normalization: LayerNorm.';
      record(s.id, s.label, ok, message);
      if (normAlt) notes.push('RMSNorm is allowed in Experiment mode. The original architecture uses LayerNorm.');
    }
    for (const lane of lanes(game)) for (const meta of modulesFor(game, lane)) {
      const m = ensureModule(game, meta.id);
      if (meta.kind === 'attention') {
        for (const step of ATTENTION_STEPS) {
          const expected = step.key === 'mask' ? meta.mask : step.expected;
          const actual = m.slots[step.key];
          let message = !actual ? `Add an operation to ${step.label.toLowerCase()}.` : `For "${step.label}", use ${BY_ID[expected].name}.`;
          if (step.key === 'mask' && actual !== expected) message = meta.mask === 'causal' ? 'Decoder self-attention must not see future tokens. Apply a causal mask before softmax.' : meta.id.endsWith('.cross') ? 'Cross-attention can see the full encoder input. Decoder self-attention masks future output tokens.' : 'The encoder needs context on both sides. A causal mask would make it autoregressive.';
          record(meta.id + '.' + step.key, step.label, actual === expected, message, meta.id);
        }
        for (const role of ['q', 'k', 'v']) {
          const expectedSource = meta.id.endsWith('.cross') && role !== 'q' ? 'encoder' : 'hidden';
          record(meta.id + '.source.' + role, `${role.toUpperCase()} source`, m.sources[role] === expectedSource, expectedSource === 'encoder' ? `${role.toUpperCase()} must come from the encoder output. Only Q comes from the decoder.` : `${role.toUpperCase()} must come from the current ${lane} hidden states.`, meta.id);
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
      if (meta.kind === 'attention') { total += ATTENTION_STEPS.length; filled += ATTENTION_STEPS.filter(s => m.slots[s.key]).length; }
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
      if (meta.kind === 'attention') {
        for (const s of ATTENTION_STEPS) m.slots[s.key] = s.key === 'mask' ? meta.mask : s.expected;
        for (const role of ['q', 'k', 'v']) m.sources[role] = meta.id.endsWith('.cross') && role !== 'q' ? 'encoder' : 'hidden';
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
        steps.push({ id: meta.id, lane, label: meta.name, desc: meta.id.endsWith('.cross') ? `Q ← decoder, K/V ← encoder. Scores: [batch, ${game.heads}, T_decoder, T_encoder].` : meta.kind === 'ffn' ? `Apply the same FFN separately to every token. Output: ${shape}.` : `${game.heads} heads × ${game.d / game.heads} features. ${meta.mask === 'causal' ? 'Future tokens are masked.' : 'All non-padding context is available.'}` });
        steps.push({ id: meta.id + '.add', lane, label: 'Residual Add', desc: `Add x and the sublayer output with matching shape ${shape}.` });
        if (game.norm === 'post') steps.push({ id: meta.id + '.norm', lane, label: 'Normalize after the residual', desc: `LayerNorm(x + F(x)), shape ${shape}.` });
      }
      for (const s of t.filter(s => s.scope === 'output')) steps.push({ id: s.id, lane, label: BY_ID[game.slots[s.id]]?.name || s.label, desc: s.id.endsWith('.probs') ? 'Softmax over the vocabulary gives next-token probabilities. This is separate from attention softmax.' : s.id.endsWith('.head') ? 'Linear: d_model → vocabulary_size.' : `Final normalization of the Pre-LN stack output: [batch, T_${lane}, ${game.d}].` });
    }
    return steps;
  }
  const api = { BLOCKS, BY_ID, COARSE_BLOCKS, COARSE_BY_ID, PROFILES, ATTENTION_STEPS, createGame, lanes, modulesFor, coarseTopology, ensureStructure, checkStructure, coarseProgress, topology, ensureModule, check, progress, fillReference, trace, clone };
  root.TransformerEngine = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
