const assert = require('node:assert/strict');
const E = require('../engine.js');
const config = require('../assets/qwen3.8-27b-config.json').text_config;
let count = 0;
const check = (condition, label) => { assert.ok(condition, label); count++; };
const game = () => E.fillReference(E.createGame('qwen'));
const fails = (g, id) => E.check(g).issues.some(i => i.id === id);
const p = E.PROFILES.qwen;
for (const [key, source] of Object.entries({d:'hidden_size',heads:'num_attention_heads',hidden:'intermediate_size',depth:'num_hidden_layers',headDim:'head_dim',kvHeads:'num_key_value_heads'})) check(p[key] === config[source], 'Profile agrees with official ' + source);
check(config.layer_types.filter(t=>t==='linear_attention').length === 48, '48 DeltaNet layers');
check(config.layer_types.filter(t=>t==='full_attention').length === 16, '16 full-attention layers');
check(config.layer_types.every((t,i)=>t===(i%4===3?'full_attention':'linear_attention')), 'Three DeltaNet then one attention per group');
check(config.output_gate_type==='swish', 'Checkpoint output gate is swish');
check(!E.check(E.createGame('qwen')).valid, 'Empty Qwen does not complete');
check(E.progress(E.createGame('qwen')).percent===0, 'Empty Qwen has zero progress');
const g = game();
check(E.check(g).valid && E.checkStructure(g).valid, 'Qwen reference completes both stages');
check(E.progress(g).percent===100, 'All internal operations counted');
check(E.coarseTopology(g).length===14, 'Both representative layers have independent macro slots');
check(E.headDim(g)===256 && g.d/g.heads!==256, 'Explicit head width independent of model width');
check(E.topology(g).filter(s=>s.scope==='embedding').map(s=>s.expected).join()==='embedding', 'Token embeddings without additive positions');
check(E.topology(g).filter(s=>s.id.endsWith('.norm')||s.id.endsWith('.finalNorm')).every(s=>s.expected==='rmsnorm'), 'RMSNorm throughout residual stack and final output');
check(E.modulesFor(g,'decoder').map(m=>m.id).join()==='decoder.delta,decoder.delta.ffn,decoder.self,decoder.ffn', 'Both FFNs follow their mixers');
for(const meta of E.modulesFor(g,'decoder')) {
 for(const step of E.moduleSteps(g,meta)) {
  const bad=game();bad.modules[meta.id].slots[step.key]=step.expected==='softmax'?'add':'softmax';
  check(fails(bad,meta.id+'.'+step.key), meta.id+': wrong '+step.key+' rejected');
 }
}
for(const [id,value] of [['decoder.self.rope','learned-position'],['decoder.self.mask','bidirectional'],['decoder.self.gateAct','sigmoid'],['decoder.delta.qkNorm','rmsnorm'],['decoder.delta.rule','qk']]) {
 const bad=game(), at=id.lastIndexOf('.');bad.modules[id.slice(0,at)].slots[id.slice(at+1)]=value;
 check(fails(bad,id), 'Reference rejects incorrect '+id);
}
for(const id of ['decoder.self.norm','decoder.delta.norm','decoder.finalNorm']) {
 const bad=game();bad.slots[id]='layernorm';check(fails(bad,id),'LayerNorm does not replace reference RMSNorm');
 bad.experimental=true;check(E.check(bad).valid,'Alternative outer norm allowed as experiment');
}
for(const id of ['decoder.delta.ffn','decoder.ffn']) {
 for(const branch of ['gateProj','upProj','downProj']) {
  const bad=game();bad.modules[id].widths[branch]++;
  check(fails(bad,id+'.width.'+branch),'Wrong gated branch width rejected');
 }
}
const wrong=game();wrong.heads=25;check(fails(wrong,'config.heads'),'Query count must divide into KV groups');
const dims=game();dims.d=768;check(fails(dims,'config.d'),'Reference model dimensions enforced');
const source=game();source.modules['decoder.self'].sources.k='encoder';check(fails(source,'decoder.self.source.k'),'Qwen self-attention K comes from input');
const trace=E.trace(g);
for(const meta of E.modulesFor(g,'decoder')) {
 const index=id=>trace.findIndex(s=>s.id===id);
 check(index(meta.id+'.norm')<index(meta.id)&&index(meta.id)<index(meta.id+'.add'),'Pre-RMSNorm execution order');
}
const legacy=game();delete legacy.structure;E.ensureStructure(legacy);check(E.check(legacy).valid,'Migration handles gated modules without serial layers');
console.log(`PASS: ${count} Qwen source, hybrid topology, gating, and validation checks.`);
