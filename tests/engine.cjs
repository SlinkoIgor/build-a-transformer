const assert = require('node:assert/strict');
const E = require('../engine.js');
let assertions = 0;
function ok(condition, message) { assert.ok(condition, message); assertions++; }
function hasIssue(game, id) { return E.check(game).issues.some(i => i.id === id); }
for (const level of ['bert', 'gpt', 'encdec']) {
  const empty = E.createGame(level);
  ok(!E.check(empty).valid, level + ': пустая схема не проходит');
  ok(E.progress(empty).percent === 0, level + ': пустой прогресс');
  for (const norm of ['post', 'pre']) {
    const game = E.createGame(level); game.norm = norm;
    if (norm !== E.PROFILES[level].norm) {
      E.fillReference(game);
      ok(hasIssue(game, 'config.norm'), `${level}/${norm}: другой порядок не считается эталоном`);
      game.experimental = true;
    }
    E.fillReference(game);
    ok(E.check(game).valid, `${level}/${norm}: эталон корректен`);
    ok(E.progress(game).percent === 100, `${level}/${norm}: заполнены все блоки`);
    const trace = E.trace(game);
    for (const lane of E.lanes(game)) {
      for (const meta of E.modulesFor(game, lane)) {
        const f = trace.findIndex(s => s.id === meta.id);
        const n = trace.findIndex(s => s.id === meta.id + '.norm');
        const a = trace.findIndex(s => s.id === meta.id + '.add');
        ok(norm === 'pre' ? n < f && f < a : f < a && a < n, `${level}/${norm}: порядок потока`);
      }
      ok(norm === 'pre' ? trace.some(s => s.id === lane + '.finalNorm') : !trace.some(s => s.id === lane + '.finalNorm'), `${level}/${norm}: финальная нормировка`);
    }
    const m = game.modules[E.lanes(game)[0] + '.ffn'];
    m.layers[2].width = game.d + 1;
    ok(hasIssue(game, E.lanes(game)[0] + '.ffn.residual'), `${level}/${norm}: residual shape mismatch`);
  }
}
const bert = E.fillReference(E.createGame('bert'));
bert.modules['encoder.self'].slots.mask = 'causal';
ok(hasIssue(bert, 'encoder.self.mask'), 'BERT не принимает causal');
bert.modules['encoder.self'].slots.mask = 'bidirectional';
bert.slots['encoder.position'] = 'rope';
ok(hasIssue(bert, 'encoder.position'), 'RoPE нельзя прибавлять к embedding');
bert.slots['encoder.position'] = 'learned-position';
bert.modules['encoder.ffn'].layers[1].type = 'relu';
ok(hasIssue(bert, 'encoder.ffn.layer.1'), 'Эталон BERT требует GELU');
bert.experimental = true;
ok(E.check(bert).valid, 'ReLU принимается в эксперименте');
bert.slots['encoder.self.norm'] = 'rmsnorm';
ok(E.check(bert).valid, 'RMSNorm принимается в эксперименте');
bert.slots['encoder.self.norm'] = 'batchnorm';
ok(hasIssue(bert, 'encoder.self.norm'), 'BatchNorm не принимается как token normalization');
bert.slots['encoder.self.norm'] = 'layernorm';
bert.modules['encoder.ffn'].layers = [
  {type:'linear',width:1536},{type:'silu',width:1536},{type:'linear',width:2048},
  {type:'tanh',width:2048},{type:'linear',width:768},
];
ok(E.check(bert).valid, 'Многослойный экспериментальный FFN');
bert.experimental = false;
ok(hasIssue(bert, 'encoder.ffn.length'), 'Глубокий FFN отличается от эталона');
bert.experimental = true;
bert.modules['encoder.ffn'].layers = [{type:'linear',width:1536},{type:'linear',width:1536},{type:'linear',width:768}];
ok(hasIssue(bert, 'encoder.ffn.nonlinearity'), 'FFN без нелинейности отклонён');
const gpt = E.fillReference(E.createGame('gpt'));
gpt.modules['decoder.self'].slots.mask = 'bidirectional';
ok(hasIssue(gpt, 'decoder.self.mask'), 'GPT не видит будущие токены');
gpt.modules['decoder.self'].slots.mask = 'causal';
gpt.heads = 7;
ok(hasIssue(gpt, 'config.heads'), 'Количество голов должно делить d_model');
const ed = E.fillReference(E.createGame('encdec'));
ok(ed.modules['decoder.cross'].sources.k === 'encoder' && ed.modules['decoder.cross'].sources.v === 'encoder', 'Cross K/V из encoder');
ed.modules['decoder.cross'].sources.k = 'hidden';
ok(hasIssue(ed, 'decoder.cross.source.k'), 'Cross K из decoder отклонён');
ed.modules['decoder.cross'].sources.k = 'encoder';
ed.modules['decoder.cross'].sources.q = 'encoder';
ok(hasIssue(ed, 'decoder.cross.source.q'), 'Cross Q из encoder отклонён');
ed.modules['decoder.cross'].sources.q = 'hidden';
ed.modules['decoder.cross'].slots.mask = 'causal';
ok(hasIssue(ed, 'decoder.cross.mask'), 'Cross-attention не использует decoder causal-mask');
ed.modules['decoder.cross'].slots.mask = 'bidirectional';
delete ed.modules['encoder.self'].slots.out;
ok(hasIssue(ed, 'encoder.self.out'), 'Выходная проекция MHA обязательна');
console.log(`PASS: ${assertions} проверок научной логики, 3 уровня × 2 порядка нормализации.`);
