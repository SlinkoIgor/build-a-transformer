const assert = require('node:assert/strict');
const E = require('../engine.js');
const config = require('../assets/mellum2-config.json');
const profile = E.PROFILES.mellum;
let count = 0;
const check = (condition, label) => { assert.ok(condition, label); count++; };
const reference = () => E.fillReference(E.createGame('mellum'));
const issue = (g, id) => E.check(g).issues.some(i => i.id === id);
for (const [key, source] of Object.entries({d:'hidden_size',heads:'num_attention_heads',kvHeads:'num_key_value_heads',headDim:'head_dim',depth:'num_hidden_layers',hidden:'moe_intermediate_size',denseHidden:'intermediate_size',experts:'num_experts',topK:'num_experts_per_tok',window:'sliding_window'})) check(profile[key]===config[source], 'Matches original '+source);
check(config.layer_types.every((type,i)=>type===(i%4===3?'full_attention':'sliding_attention')), 'Three local layers then one full layer');
check(config.layer_types.filter(t=>t==='sliding_attention').length===21, '21 local-attention layers');
check(config.layer_types.filter(t=>t==='full_attention').length===7, '7 full-attention layers');
check(config.mlp_layer_types.every(t=>t==='sparse'), 'Every layer uses sparse experts');
check(config.norm_topk_prob, 'Selected expert probabilities are normalized');
check(config.rope_parameters.full_attention.rope_type==='yarn' && config.rope_parameters.full_attention.factor===16, 'YaRN for full attention');
check(config.rope_parameters.sliding_attention.rope_type==='default', 'Ordinary RoPE for local attention');
const game=reference();
check(E.check(game).valid, 'Mellum reference passes');
check(E.progress(game).percent===100, 'All nested operations count toward completion');
check(!E.check(E.createGame('mellum')).valid, 'Empty Mellum is incomplete');
check(E.coarseTopology(game).length===14, 'Two representative layers on the full canvas');
check(E.coarseTopology(game).filter(s=>s.expected==='mixture-of-experts').length===2, 'MoE is a whole module');
check(E.topology(game).filter(s=>s.scope==='embedding').length===1, 'No additive position vectors');
check(E.headDim(game)===128 && game.d/game.heads!==128, 'Explicit head dimension independent of hidden size');
for (const meta of E.modulesFor(game,'decoder')) {
 for (const step of E.moduleSteps(game,meta)) {
  const g=reference();g.modules[meta.id].slots[step.key]=step.expected==='add'?'linear':'add';
  check(issue(g,meta.id+'.'+step.key), 'Incorrect '+meta.id+'.'+step.key+' rejected');
 }
 if(meta.kind==='attention') {
  check(!E.moduleSteps(game,meta).some(s=>s.key==='gateAct'), 'Attention has no output gate');
  const g=reference();g.modules[meta.id].sources.v='encoder';check(issue(g,meta.id+'.source.v'),'V source must be current input');
 }
 if(meta.kind==='moe') {
  const g=reference();g.modules[meta.id].routing.topK=4;check(issue(g,meta.id+'.routing.topK'),'Reference selects eight experts');
  g.modules[meta.id].routing.topK=8;g.modules[meta.id].routing.experts=32;check(issue(g,meta.id+'.experts'),'Reference has 64 experts');
  for(const branch of ['gateProj','upProj','downProj']) {
   const bad=reference();bad.modules[meta.id].widths[branch]=7168;check(issue(bad,meta.id+'.width.'+branch),'Dense width is not expert width');
  }
 }
}
const local=reference();local.modules['decoder.local'].slots.mask='causal';check(issue(local,'decoder.local.mask'),'Unrestricted causal mask rejected in sliding layer');
const full=reference();full.modules['decoder.self'].slots.mask='sliding-causal';check(issue(full,'decoder.self.mask'),'Windowed mask rejected in full causal layer');
full.modules['decoder.self'].slots.mask='causal';full.modules['decoder.self'].slots.rope='rope';check(issue(full,'decoder.self.rope'),'Full attention requires YaRN-scaled RoPE');
const plain=reference();plain.structure['decoder.ffn']='feed-forward';check(issue(plain,'structure.decoder.ffn'),'Dense FFN cannot replace sparse experts');
const wrongDim=reference();wrongDim.heads=30;check(issue(wrongDim,'config.heads'),'Head groups must divide query heads');
const wrongNorm=reference();wrongNorm.slots['decoder.self.norm']='layernorm';check(issue(wrongNorm,'decoder.self.norm'),'Reference needs RMSNorm');
const legacy=reference();delete legacy.structure;E.ensureStructure(legacy);check(E.check(legacy).valid,'Sparse modules survive save migration');
const trace=E.trace(game);
for(const meta of E.modulesFor(game,'decoder')) {
 const i=id=>trace.findIndex(s=>s.id===id);check(i(meta.id+'.norm')<i(meta.id)&&i(meta.id)<i(meta.id+'.add'),'Pre-RMSNorm residual order');
}
console.log(`PASS: ${count} Mellum source, routing, window, RoPE, and validation checks.`);
