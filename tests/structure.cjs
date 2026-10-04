const assert = require('node:assert/strict');
const E = require('../engine.js');
let n=0;
const check=(condition,label)=>{assert.ok(condition,label);n++;};
for(const [level,count,year] of [['bert',5,2018],['gpt',8,2019],['encdec',13,2017]]) {
 const g=E.createGame(level);
 check(E.PROFILES[level].year===year,level+' year');
 check(E.coarseProgress(g).total===count,level+' module count');
 check(!E.checkStructure(g).valid,'empty architecture rejected');
 for(const s of E.coarseTopology(g))g.structure[s.id]=s.expected;
 check(E.checkStructure(g).valid,level+' macro-only architecture valid');
 check(!E.check(g).valid,level+' macro-only model incomplete');
 check(Object.keys(g.slots).length===0,level+' no automatic internal operations');
 E.fillReference(g);
 check(E.check(g).valid,level+' reference still complete');
 const module=E.lanes(g)[0]+'.self';
 const internal=JSON.stringify(g.modules[module]);
 g.structure[module]='recurrent';
 check(!E.checkStructure(g).valid&&!E.check(g).valid,level+' wrong macro rejected despite correct internals');
 delete g.structure[module];
 check(JSON.stringify(g.modules[module])===internal,level+' internal work survives removal');
 g.structure[module]='self-attention';
 if(level==='gpt')g.structure[module]='masked-self-attention';
 check(E.check(g).valid,level+' restored macro recovers complete model');
 const legacy=E.clone(g);delete legacy.structure;
 E.ensureStructure(legacy);
 check(E.check(legacy).valid,level+' legacy save migrated');
 check(JSON.stringify(legacy.slots)===JSON.stringify(g.slots)&&JSON.stringify(legacy.modules)===JSON.stringify(g.modules),level+' migration preserves internals');
 const emptyLegacy=E.createGame(level);delete emptyLegacy.structure;E.check(emptyLegacy);
 check(E.coarseProgress(emptyLegacy).filled===0,level+' empty legacy save stays empty');
 for(const norm of ['pre','post']) {
  const experiment=E.createGame(level);experiment.norm=norm;experiment.experimental=true;
  E.fillReference(experiment);
  check(E.checkStructure(experiment).valid&&E.check(experiment).valid,level+'/'+norm+' experiment');
  for(const s of E.coarseTopology(experiment).filter(s=>/\.(residual|norm|add)$/.test(s.id))) {
   experiment.structure[s.id]=s.expected==='normalization'?'residual-add':'normalization';
   check(E.checkStructure(experiment).issues.some(i=>i.id==='structure.'+s.id),level+'/'+norm+' wrapper order enforced');
   experiment.structure[s.id]=s.expected;
  }
 }
}
for(const level of ['gpt','bert','encdec']) {
 const g=E.createGame(level);g.norm='pre';g.experimental=level!=='gpt';E.fillReference(g);
 const slots=JSON.stringify(g.slots), modules=JSON.stringify(g.modules);
 for(const lane of E.lanes(g))for(const meta of E.modulesFor(g,lane)) {
  delete g.structure[meta.id+'.norm'];delete g.structure[meta.id+'.add'];g.structure[meta.id+'.residual']='pre-residual';
 }
 E.ensureStructure(g);
 check(E.check(g).valid,level+' older Pre-LN wrapper save migrates into a valid sequence');
 check(JSON.stringify(g.slots)===slots&&JSON.stringify(g.modules)===modules,level+' migration keeps every internal operation');
 check(!Object.values(g.structure).includes('pre-residual'),level+' migration removes the combined wrapper');
 delete g.slots[E.lanes(g)[0]+'.self.norm'];
 check(!E.check(g).valid,level+' migration never invents an unbuilt normalization operation');
}
const partial=E.createGame('gpt');partial.structure={'decoder.input':'input-embeddings','decoder.self':'masked-self-attention','decoder.self.residual':'pre-residual'};
E.ensureStructure(partial);
check(E.coarseProgress(partial).filled===4&&!E.checkStructure(partial).valid,'Partial older save keeps the unbuilt FFN empty');
check(Object.keys(partial.slots).length===0,'Partial save migration does not fill individual operations');
console.log(`PASS: ${n} architecture, migration, and two-stage validation checks.`);
