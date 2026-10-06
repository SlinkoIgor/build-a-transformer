(() => {
  'use strict';
  const E = window.TransformerEngine;
  const { BLOCKS, COARSE_BLOCKS, PROFILES, ATTENTION_STEPS } = E;
  const BY_ID = { ...E.BY_ID, ...E.COARSE_BY_ID };
  const PAPERS = window.TransformerPapers;
  const $ = selector => document.querySelector(selector);
  const escape = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
  const icons = {
    check: '<svg viewBox="0 0 20 20" fill="none"><path d="m4 10 4 4 8-8" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    close: '<svg viewBox="0 0 20 20" fill="none"><path d="m5 5 10 10M5 15 15 5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
    undo: '<svg viewBox="0 0 20 20" fill="none"><path d="M7 4 3 8l4 4M3 8h8a5 5 0 0 1 0 10" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  };
  const groupNames = { input: 'Input & positions', attention: 'Attention operations', norm: 'Addition & normalization', ffn: 'Activations', output: 'Output modules' };
  let session = loadSession(), game = session.games[session.active];
  let selected = null, activeTarget = null, search = '', filter = 'all', dialog = null;
  let phase = 'input', assembly = game.assemblyStage || 'architecture', feedback = null, moduleFeedback = null;
  let histories = {}, toastTimer, traceTimer, traceState = null, maskQuery = 2, draggingLayer = null, returnFocus = null;
  let renderedLevel = null, renderedPhase = null, renderedDialogKey = null;
  let projector = false;
  try { projector = localStorage.getItem('transformer-text-size-v2') === 'large'; } catch (_) {}
  function loadSession() {
    try {
      const data = JSON.parse(localStorage.getItem('transformer-lab-v1'));
      if (data?.version === 1 && PROFILES[data.active] && ['bert', 'gpt', 'encdec'].every(k => data.games[k]?.level === k)) {
        for (const key of Object.keys(PROFILES)) data.games[key] ||= E.createGame(key);
        Object.values(data.games).forEach(E.ensureStructure);
        return data;
      }
    } catch (_) {}
    return { version: 1, active: 'bert', games: Object.fromEntries(Object.keys(PROFILES).map(k => [k, E.createGame(k)])), achievements: {} };
  }
  function persist() { try { localStorage.setItem('transformer-lab-v1', JSON.stringify(session)); } catch (_) {} }
  function remember() {
    const h = histories[game.level] ||= []; h.push(E.clone(game)); if (h.length > 60) h.shift();
    game.completed = false; feedback = null; moduleFeedback = null; stopTrace();
  }
  function toast(message) {
    $('#toast').textContent = message; $('#toast').classList.add('visible'); clearTimeout(toastTimer);
    toastTimer = setTimeout(() => $('#toast').classList.remove('visible'), 5000);
  }
  function captureScroll(selectors) {
    return selectors.map(selector => { const el = $(selector); return { selector, top: el?.scrollTop || 0, left: el?.scrollLeft || 0 }; });
  }
  function restoreScroll(items) { for (const item of items) { const el = $(item.selector); if (el) { el.scrollTop = item.top; el.scrollLeft = item.left; } } }
  function focusIdentity() {
    const el = document.activeElement;
    for (const key of ['target', 'source', 'width', 'config', 'block', 'norm', 'phase', 'stage', 'action']) if (el?.dataset?.[key] !== undefined) return { key, value: el.dataset[key] };
    return null;
  }
  function restoreFocus(identity) {
    if (!identity) return;
    const scope = dialog ? $('.dialog') : $('#app');
    [...scope.querySelectorAll('button,input,select')].find(el => el.dataset[identity.key] === identity.value)?.focus({ preventScroll: true });
  }
  function phases() {
    return [
      { id: 'input', label: 'Input', desc: assembly === 'architecture' ? 'Place the input module. You will build its embedding operations in step 2.' : 'Turn tokens into input vectors. Combine their embeddings.' },
      ...E.lanes(game).map(lane => ({ id: lane, label: lane === 'encoder' ? 'Encoder block' : 'Decoder block', desc: assembly === 'architecture' ? `Choose whole modules for one ${lane} block.` : `Build one ${lane} block. Operations flow downward; sublayers run from left to right.` })),
      { id: 'output', label: 'Output', desc: game.level === 'bert' ? 'The encoder returns one contextual vector per input token.' : 'Convert the final decoder states into next-token probabilities.' },
    ];
  }
  function targetPhase(id) {
    if (id.startsWith('structure.')) id = id.slice('structure.'.length);
    const stage = E.topology(game).find(s => s.id === id);
    if (stage?.scope === 'embedding' || /\.input$/.test(id)) return 'input';
    if (stage?.scope === 'output' || /\.output$/.test(id)) return 'output';
    if (/^(encoder|decoder)\.(self|cross|ffn|delta|local)/.test(id)) return id.split('.')[0];
    return 'input';
  }
  function phaseProgress(id) {
    const result = assembly === 'architecture' ? E.checkStructure(game) : E.check(game);
    const matches = result.checks.filter(c => !c.id.startsWith('config.') && targetPhase(c.module || c.id) === id);
    return { done: matches.filter(c => c.ok).length, total: matches.length };
  }
  function render() {
    const view = assembly;
    const scroll = renderedLevel === game.level && renderedPhase === view ? captureScroll(['#canvas', '#palette-content']) : [];
    const focus = focusIdentity(), p = PROFILES[game.level], prog = assembly === 'architecture' ? E.coarseProgress(game) : E.progress(game);
    const structureReady = E.checkStructure(game).valid;
    document.documentElement.classList.toggle('projector', projector);
    $('#app').innerHTML = `
      <nav class="level-navigation" aria-label="Choose a model"><div class="level-tabs">${Object.entries(PROFILES).map(([key, profile]) => `<button class="level-tab ${game.level === key ? 'active' : ''}" data-action="level" data-level="${key}" aria-current="${game.level === key ? 'page' : 'false'}">${session.achievements[key] ? `<span class="level-number">${icons.check}</span>` : ''}${profile.name}<span class="model-year">${profile.year}</span></button>`).join('')}</div><div class="header-actions"><button class="text-button" data-action="reference">Papers</button><button class="text-button" data-action="settings">Settings</button><button class="text-button" data-action="guide">Help</button><button class="text-button" data-action="fullscreen">${document.fullscreenElement ? 'Exit fullscreen' : 'Fullscreen'}</button></div></nav>
      <main class="lab-layout"><aside class="palette-panel" aria-label="Building blocks"><div class="panel-heading"><h2>${assembly === 'architecture' ? 'Modules' : 'Building blocks'}</h2></div><label class="search-box"><input id="block-search" type="search" placeholder="Search blocks…" value="${escape(search)}" aria-label="Search blocks"></label><div class="palette-filters" role="group" aria-label="Block categories">${[['all', 'All'], ['input', 'Input'], ['attention', 'Attention'], ['norm', 'Norm'], ['ffn', 'FFN'], ...(assembly === 'architecture' ? [['output', 'Output']] : [])].map(([id, label]) => `<button data-action="filter" data-filter="${id}" class="${filter === id ? 'active' : ''}" aria-pressed="${filter === id}">${label}</button>`).join('')}</div><div id="palette-content" class="palette-content">${paletteHtml()}</div><div class="selection-info" aria-live="polite">${selectionHtml()}</div></aside>
        <section class="workspace" aria-label="Transformer assembly"><div class="workspace-heading"><h1 class="visually-hidden">${p.label} ${p.year}</h1><div class="assembly-stages" role="group" aria-label="Build view"><button data-action="stage" data-stage="architecture" class="${assembly === 'architecture' ? 'active' : ''}" aria-pressed="${assembly === 'architecture'}">Architecture</button><button data-action="stage" data-stage="inside" class="${assembly === 'inside' ? 'active' : ''}" aria-pressed="${assembly === 'inside'}" ${structureReady ? '' : 'disabled'} title="${structureReady ? 'Build the operations inside your modules' : 'Complete the architecture first'}">Inside blocks</button></div><div class="canvas-meta"><span>${PROFILES[game.level].normType === 'rmsnorm' ? (game.norm === 'pre' ? 'Pre-RMSNorm' : 'Post-RMSNorm') : game.norm === 'pre' ? 'Pre-LN' : 'Post-LN'}${game.experimental ? ' · Experiment' : ''}</span><span>Flow ↑ bottom to top</span></div></div>
          <div class="canvas" id="canvas">${networkHtml()}${feedback ? feedbackHtml(feedback) : ''}</div>
          <footer class="workspace-footer"><div class="assembly-count"><strong>${prog.filled} / ${prog.total} ${assembly === 'architecture' ? 'modules' : 'blocks'}</strong><div><button class="text-button" data-action="undo" ${histories[game.level]?.length ? '' : 'disabled'}>${icons.undo}Undo</button><button class="text-button" data-action="hint">Hint</button></div></div><div class="footer-actions"><button class="secondary-button" data-action="show-solution">Show correct solution</button>${assembly === 'architecture' ? `<button class="secondary-button" data-action="stage" data-stage="inside" ${structureReady ? '' : 'disabled'}>Build inside ↗</button>` : '<button class="secondary-button" data-action="trace">Trace shapes</button>'}<button class="primary-button" data-action="validate">${icons.check}${assembly === 'architecture' ? 'Check architecture' : 'Check build'}</button></div></footer></section></main>`;
    if (dialog) renderDialog(false);
    restoreScroll(scroll); restoreFocus(focus); drawResiduals(); drawNetworkConnections(); renderedLevel = game.level; renderedPhase = view;
  }

  function drawResiduals() {
    document.querySelectorAll('.residual-flow').forEach(flow => {
      const add = flow.querySelector('[data-target$=".add"]'), line = flow.querySelector('.residual-bypass');
      if (!add || !line) return;
      const a = add.getBoundingClientRect(), f = flow.getBoundingClientRect();
      line.style.height = a.top - f.top + a.height / 2 + 'px';
    });
  }
  window.addEventListener('resize', drawResiduals);
  function selectionHtml() {
    if (activeTarget) return '<strong>Slot selected</strong><p>Choose a block to place here.</p>';
    return selected ? `<strong>${BY_ID[selected].name}</strong><p>${BY_ID[selected].desc}</p>` : '<p>Select or drag a block.</p>';
  }
  function paletteHtml() {
    const result = (assembly === 'architecture' ? COARSE_BLOCKS : BLOCKS).filter(b => (filter === 'all' || b.group === filter || (filter === 'ffn' && b.id === 'linear')) && (b.name + ' ' + b.desc).toLowerCase().includes(search.toLowerCase()));
    if (!result.length) return '<p class="empty-search">No matching blocks.</p>';
    const order = assembly === 'architecture' ? ['input', 'attention', 'ffn', 'norm', 'output'] : phase === 'input' ? ['input', 'norm', 'attention', 'ffn', 'output'] : ['norm', 'attention', 'ffn', 'input', 'output'];
    const names = assembly === 'architecture' ? { input: 'Input modules', attention: 'Attention modules', norm: 'Normalization & Add', ffn: 'Feed-forward modules', output: 'Output modules' } : groupNames;
    return order.map(group => { const list = result.filter(b => b.group === group); return list.length ? `<section class="block-group"><h3>${names[group]}</h3>${list.map(libraryBlock).join('')}</section>` : ''; }).join('');
  }
  function libraryBlock(b) { return `<button class="library-block ${b.group} ${selected === b.id ? 'selected' : ''}" draggable="true" data-action="select" data-block="${b.id}" title="${escape(b.desc)}" aria-pressed="${selected === b.id}"><span class="block-symbol">${b.symbol}</span><span class="block-name">${b.name}</span></button>`; }
  function blockChip(id) { const b = BY_ID[id]; return b ? `<span class="block-chip ${b.group}"><span class="block-symbol">${b.symbol}</span><span>${b.name}</span></span>` : ''; }
  function slotHtml(id, options = {}) {
    const value = options.value === undefined ? game.slots[id] : options.value;
    const issue = (moduleFeedback?.issues || feedback?.issues || []).find(i => i.id === id);
    const stage = E.topology(game).find(s => s.id === id), label = options.bare ? 'Choose a block' : options.label || stage?.label || 'Operation', target = options.target || id;
    return `<div class="slot-wrap ${issue ? 'has-error' : ''} ${traceState?.steps[traceState.index]?.id === id ? 'trace-active' : ''}" data-node="${escape(id)}">${options.bare ? '' : `<span class="slot-role">${escape(label)}</span>`}<div class="slot-control"><button class="drop-slot ${value ? 'filled ' + BY_ID[value]?.group : 'empty'} ${selected ? 'ready' : ''} ${activeTarget === target ? 'slot-selected' : ''}" data-action="slot" data-target="${escape(target)}" aria-label="${escape(label + ': ' + (BY_ID[value]?.name || 'empty slot'))}" title="${escape(value ? BY_ID[value]?.desc : label)}">${value ? blockChip(value) : '<span class="slot-plus">+</span><span>Choose a block</span>'}</button>${value ? `<button class="remove-slot" data-action="remove" data-target="${escape(target)}" aria-label="Remove ${escape(BY_ID[value]?.name)}" title="Remove block">×</button>` : ''}</div>${options.caption ? `<span class="slot-caption">${options.caption}</span>` : ''}</div>`;
  }

  const wire = () => '<div class="wire" aria-hidden="true">↓</div>';
  function phaseHtml() { return networkHtml(); }

  function insideProgress(id) {
    let ids, values;
    if (id.endsWith('.input') || id.endsWith('.output')) {
      ids = E.topology(game).filter(s => s.lane === id.split('.')[0] && s.scope === (id.endsWith('.input') ? 'embedding' : 'output')).map(s => s.id);
      values = ids.map(id => game.slots[id]);
    } else if (/\.(norm|add)$/.test(id)) {
      values = [game.slots[id]];
    } else if (id.endsWith('.residual')) {
      ids = [id.replace(/\.residual$/, '.norm'), id.replace(/\.residual$/, '.add')];
      values = ids.map(id => game.slots[id]);
    } else {
      const m = E.ensureModule(game, id);
      const meta = E.modulesFor(game, id.split('.')[0]).find(m => m.id === id), steps = E.moduleSteps(game, meta);
      values = steps ? steps.map(s => m.slots[s.key]) : m.layers.map(l => l.type);
    }
    return { filled: values.filter(Boolean).length, total: values.length };
  }
  function coarseSlotHtml(id) {
    const stage = E.coarseTopology(game).find(s => s.id === id), value = E.ensureStructure(game)[id];
    const target = 'structure::' + id, issue = feedback?.issues.some(i => i.id === 'structure.' + id);
    const p = insideProgress(id), ready = E.checkStructure(game).valid;
    const open = value ? `<button class="inside-progress" data-action="open-inside" data-detail="${id}" ${ready ? '' : 'disabled'} aria-label="Build inside ${escape(BY_ID[value]?.name)}: ${p.filled} of ${p.total} operations placed" title="${ready ? 'Build inside' : 'Complete the architecture first'}"><span>${p.filled}/${p.total}</span> ↗</button>` : '';
    return `<div class="coarse-slot slot-wrap ${issue ? 'has-error' : ''}" data-node="structure.${id}"><div class="slot-control ${value ? 'module-filled ' + BY_ID[value]?.group : ''}"><button class="drop-slot ${value ? 'filled ' + BY_ID[value]?.group : 'empty'} ${activeTarget === target ? 'slot-selected' : ''}" data-action="slot" data-target="${target}" aria-label="${stage.label}: ${value ? BY_ID[value]?.name : 'empty block'}" title="${value ? escape(BY_ID[value]?.desc) : 'Choose a block'}">${value ? blockChip(value) : '<span class="slot-plus">+</span><span>Choose a block</span>'}</button>${open}${value ? `<button class="remove-slot" data-action="remove" data-target="${target}" aria-label="Remove ${escape(BY_ID[value]?.name)}" title="Remove block">×</button>` : ''}</div></div>`;
  }
  const upWire = () => '<div class="network-wire" aria-hidden="true"><span></span></div>';
  function networkModuleHtml(meta) {
    const p = insideProgress(meta.id), issue = feedback?.issues.some(i => i.module === meta.id);
    return `<button class="network-module ${['ffn', 'moe'].includes(meta.kind) ? 'ffn' : 'attention'} ${issue ? 'module-error' : ''}" data-action="module" data-module="${meta.id}" data-node="${meta.id}" aria-label="Build inside ${meta.name}"><span class="module-symbol">${['ffn', 'moe'].includes(meta.kind) ? '▥' : '✳'}</span><strong>${meta.name}</strong><span class="module-count">${p.filled}/${p.total}</span><span class="module-expand" aria-hidden="true">↗</span></button>`;
  }
  function networkSublayerHtml(meta) {
    const operation = assembly === 'architecture' ? coarseSlotHtml(meta.id) : networkModuleHtml(meta);
    const node = id => assembly === 'architecture' ? coarseSlotHtml(id) : slotHtml(id, { bare: true });
    const top = game.norm === 'pre' ? meta.id + '.add' : assembly === 'architecture' ? meta.id + '.residual' : meta.id + '.norm';
    const content = game.norm === 'pre' ? node(top) + upWire() + operation + upWire() + node(meta.id + '.norm') : assembly === 'architecture' ? node(top) + upWire() + operation : node(top) + upWire() + node(meta.id + '.add') + upWire() + operation;
    return `<div class="network-sublayer" data-sublayer="${meta.id}"><svg class="network-bypass" aria-hidden="true"></svg><div class="network-sublayer-main">${content}</div></div>`;
  }
  function networkInputHtml(lane) {
    if (assembly === 'architecture') return coarseSlotHtml(lane + '.input');
    const stages = E.topology(game).filter(s => s.lane === lane && s.scope === 'embedding');
    const base = stages.filter(s => !/\.(embedAdd|embedNorm)$/.test(s.id));
    const combined = stages.filter(s => /\.(embedAdd|embedNorm)$/.test(s.id)).reverse();
    const fan = base.map((_, i) => {
      const x = (i + .5) / base.length * 100;
      return `M${x} 13 V7 H50 V0`;
    }).join(' ');
    return `<div class="network-input">${combined.map(s => slotHtml(s.id, { bare: true })).join(upWire())}<svg class="embedding-fan" viewBox="0 0 100 14" preserveAspectRatio="none" aria-hidden="true"><path d="${fan}"/><path class="fan-arrow" d="M48 3 L50 0 L52 3"/></svg><div class="network-embedding-branches">${base.map(s => slotHtml(s.id, { bare: true })).join('')}</div></div>`;
  }
  function networkLaneHtml(lane) {
    const name = lane === 'encoder' ? 'Encoder' : 'Decoder', modules = E.modulesFor(game, lane).slice().reverse();
    const outputs = assembly === 'architecture' ? E.coarseTopology(game).filter(s => s.lane === lane && s.scope === 'output').reverse().map(s => coarseSlotHtml(s.id)) : E.topology(game).filter(s => s.lane === lane && s.scope === 'output').reverse().map(s => slotHtml(s.id, { bare: true }));
    if (PROFILES[game.level].hybridGroup) {
      const p = PROFILES[game.level], group = p.hybridGroup, all = E.modulesFor(game, lane);
      const stack = (list, label, count) => `<section class="network-stack" aria-label="${label}"><header><h2>${label}</h2><span class="repeat-count">× ${count}</span></header>${list.slice().reverse().map(networkSublayerHtml).join(upWire())}</section>`;
      return `<article class="network-lane decoder hybrid-lane" aria-label="${escape(p.name)} text decoder architecture"><div class="lane-output">Next-token probabilities</div>${upWire()}${outputs.join(upWire())}${upWire()}<section class="hybrid-group"><header><h2>Hybrid group</h2><span class="repeat-count">× ${group.repeats} · ${game.depth} layers</span></header>${stack(all.slice(2), 'Layer 4', 1)}${upWire()}${stack(all.slice(0, 2), 'Layers 1–' + group.first, group.first)}</section>${upWire()}${networkInputHtml(lane)}${upWire()}<div class="lane-input">Text tokens</div></article>`;
    }
    return `<article class="network-lane ${lane}" aria-label="${name} architecture"><div class="lane-output">${lane === 'encoder' ? 'Contextual representations' : 'Next-token probabilities'}</div>${upWire()}${outputs.length ? outputs.join(upWire()) + upWire() : ''}<section class="network-stack" aria-label="${name} block repeated ${game.depth} times"><header><h2>${name} block</h2><span class="repeat-count" title="Repeat the block ${game.depth} times">× ${game.depth}</span></header>${modules.map(networkSublayerHtml).join(upWire())}</section>${upWire()}${networkInputHtml(lane)}${upWire()}<div class="lane-input">${lane === 'encoder' ? 'Input tokens' : game.level === 'encdec' ? 'Target tokens · shifted right' : 'Previous tokens'}</div></article>`;
  }
  function networkHtml() {
    const paired = E.lanes(game).length > 1;
    return `<div class="network-board ${paired ? 'paired' : ''}">${paired ? '<svg class="cross-memory-wire" aria-hidden="true"></svg>' : ''}${E.lanes(game).map(networkLaneHtml).join('')}</div>`;
  }
  function drawNetworkConnections() {
    document.querySelectorAll('.network-sublayer').forEach(group => {
      const svg = group.querySelector('.network-bypass'), main = group.querySelector('.network-sublayer-main');
      const g = group.getBoundingClientRect(), m = main.getBoundingClientRect();
      const add = group.querySelector('[data-node$=".add"]') || group.querySelector('[data-node$=".residual"]');
      const target = add || group.querySelector('[data-node$=".norm"]');
      if (!target || !g.width || !g.height) return;
      const a = target.getBoundingClientRect(), x = m.left - g.left, y = a.top - g.top + a.height / 2, bottom = g.height - 1;
      svg.setAttribute('viewBox', `0 0 ${g.width} ${g.height}`);
      svg.innerHTML = `<path d="M${x} ${bottom} H3 V${y} H${x}"/><path d="M${x-4} ${y-3} L${x} ${y} L${x-4} ${y+3}"/>`;
    });
    const board = $('.network-board.paired'), svg = board?.querySelector('.cross-memory-wire');
    const source = board?.querySelector('.encoder .lane-output'), target = board?.querySelector('[data-sublayer="decoder.cross"] [data-node="structure.decoder.cross"], [data-sublayer="decoder.cross"] [data-node="decoder.cross"]');
    if (!board || !svg || !source || !target) return;
    const g = board.getBoundingClientRect(), a = source.getBoundingClientRect(), b = target.getBoundingClientRect();
    const left = board.querySelector('.encoder').getBoundingClientRect(), right = board.querySelector('.decoder').getBoundingClientRect();
    const stacked = right.top > left.bottom;
    const x1 = stacked ? a.left - g.left - 3 : a.right - g.left + 4, y1 = a.top - g.top + a.height / 2, mid = stacked ? 3 : (left.right + right.left) / 2 - g.left;
    const x2 = b.left - g.left - 3, y2 = b.top - g.top + b.height / 2;
    svg.setAttribute('viewBox', `0 0 ${g.width} ${g.height}`);
    svg.innerHTML = `<path class="memory-path" d="M${x1} ${y1} H${mid} V${y2} H${x2}"/><circle cx="${x1}" cy="${y1}" r="3"/><path d="M${x2-5} ${y2-4} L${x2} ${y2} L${x2-5} ${y2+4}"/><text x="${stacked ? mid+5 : mid-5}" y="${y2-9}" text-anchor="${stacked ? 'start' : 'end'}">K/V</text>`;
  }
  window.addEventListener('resize', drawNetworkConnections);

  function setAssembly(stage, target = null) {
    if (stage === 'inside' && !E.checkStructure(game).valid) { feedback = E.checkStructure(game); render(); $('.feedback')?.scrollIntoView({ block: 'nearest' }); return; }
    stopTrace(); if (dialog) closeDialog();
    if (target) phase = target;
    else if (assembly !== stage && stage === 'inside') phase = 'input';
    assembly = stage; game.assemblyStage = stage; selected = null; activeTarget = null; filter = 'all'; search = ''; feedback = null;
    persist(); render();
    if (stage === 'inside' && (!target || target === 'input')) $('#canvas').scrollTop = $('#canvas').scrollHeight;
  }
  function openInside(id) {
    if (!E.checkStructure(game).valid) return;
    setAssembly('inside', targetPhase(id));
    if (id.endsWith('.residual')) openDialog('residual', id.replace(/\.residual$/, ''));
    else if (/\.(norm|add)$/.test(id)) openDialog('operation', id);
    else if (!id.endsWith('.input') && !id.endsWith('.output')) openDialog('module', id);
  }
  function inputHtml(lane) {
    const emb = E.topology(game).filter(s => s.lane === lane && s.scope === 'embedding');
    const basic = emb.filter(s => !s.id.endsWith('embedAdd') && !s.id.endsWith('embedNorm'));
    const tokens = lane === 'encoder' && game.level === 'encdec' ? ['the', 'cat', 'sits', 'here'] : PROFILES[game.level].tokens;
    return `<article class="input-card"><h3>${lane === 'encoder' ? 'Encoder input' : 'Decoder input'}</h3><div class="input-tokens" data-node="${lane}.input">${tokens.map(t => `<span>${escape(t)}</span>`).join('')}</div><p class="input-caption">${lane === 'decoder' && game.level === 'encdec' ? 'Target tokens shifted right (teacher forcing)' : 'Token IDs → embedding vectors'}</p><div class="embedding-branches ${basic.length === 3 ? 'three' : ''}">${basic.map(s => slotHtml(s.id)).join('')}</div>${wire()}<div class="embedding-combine">${slotHtml(lane + '.embedAdd')}${game.level === 'bert' ? wire() + slotHtml(lane + '.embedNorm') : ''}</div>${wire()}<div class="flow-endpoint">To ${lane} block · [B, T, ${game.d}]</div></article>`;
  }
  function moduleHtml(meta) {
    const m = E.ensureModule(game, meta.id), steps = E.moduleSteps(game, meta), total = steps ? steps.length : m.layers.length;
    const filled = steps ? steps.filter(s => m.slots[s.key]).length : m.layers.filter(l => l.type).length;
    const errors = feedback?.issues.some(i => i.module === meta.id);
    return `<button class="module-card ${meta.kind} ${errors ? 'module-error' : ''} ${filled === total ? 'module-built' : ''}" data-action="module" data-module="${meta.id}" data-node="${meta.id}" aria-label="${meta.name}: ${filled === total ? 'open and edit' : 'build inside'}"><span class="module-tag">${meta.id.endsWith('.cross') ? 'Q: decoder · K/V: encoder' : meta.kind === 'attention' ? (meta.mask === 'causal' ? 'Causal attention' : 'Bidirectional attention') : 'Applied to each token'}</span><span class="module-count">${filled} / ${total} operations placed</span><span class="module-open">${filled === total ? 'Open & edit' : 'Build inside'} →</span></button>`;
  }
  function sublayerHtml(meta, i) {
    const norm = slotHtml(meta.id + '.norm'), add = slotHtml(meta.id + '.add');
    return `<article class="sublayer-card"><h3><span>${i + 1}</span>${meta.name}</h3><div class="flow-endpoint">Input x</div><div class="residual-flow"><div class="residual-bypass" aria-label="Residual connection bypasses the sublayer"><span>x</span></div><div class="sublayer-main">${wire()}${game.norm === 'pre' ? norm + wire() : ''}${moduleHtml(meta)}${wire()}${add}${game.norm === 'post' ? wire() + norm : ''}</div></div>${wire()}<div class="flow-endpoint">Output → ${['ffn', 'moe'].includes(meta.kind) ? 'next block' : 'next sublayer'}</div></article>`;
  }
  function blockHtml(lane) {
    const modules = E.modulesFor(game, lane);
    return `<div class="block-summary"><span>Repeat <strong>× ${game.depth}</strong> · ${game.norm === 'pre' ? 'Pre-LN' : 'Post-LN'}</span><span>Sublayers: left → right · Operations: top → bottom</span></div>${lane === 'decoder' && game.level === 'encdec' ? '<div class="memory-connection"><strong>Encoder output → cross-attention K & V</strong></div>' : ''}<div class="sublayer-grid ${modules.length === 3 ? 'three-sublayers' : ''}">${modules.map(sublayerHtml).join('')}</div><p class="build-note">The line on the left carries the original x to the residual Add. Attention and Feed Forward open into individual operations.</p>`;
  }
  function outputHtml(lane) {
    const stages = E.topology(game).filter(s => s.lane === lane && s.scope === 'output');
    return `<article class="output-card"><h3>${lane === 'encoder' ? 'Encoder output' : 'Decoder output'}</h3><div class="flow-endpoint">After ${game.depth} ${lane} blocks</div>${wire()}${stages.map(s => slotHtml(s.id) + wire()).join('')}<div class="flow-endpoint output-endpoint" data-node="${lane}.output">${lane === 'encoder' ? 'Contextual token vectors' : 'Next-token probabilities'}</div><p>${lane === 'encoder' ? game.level === 'bert' ? 'One vector per token. Task-specific BERT heads are outside this build.' : 'These states become the keys and values in decoder cross-attention.' : game.level === 'gpt' ? 'Vocabulary projection → softmax. GPT-2 shares projection weights with its token embeddings.' : 'Vocabulary projection → softmax over the target vocabulary.'}</p></article>`;
  }
  function feedbackHtml(result) {
    if (result.valid && result.kind === 'structure') return '<div class="feedback success"><strong>Architecture is correct</strong><p>Now build the operations inside your modules.</p><button class="primary-button" data-action="stage" data-stage="inside">Build inside →</button></div>';
    if (result.valid) return `<div class="feedback success"><strong>Build complete</strong><p>${game.experimental ? 'The experiment passed structure and dimension checks.' : 'All operations, masks, and dimensions are correct.'}</p>${result.notes.map(n => `<p>${escape(n)}</p>`).join('')}</div>`;
    return `<div class="feedback warning"><strong>${result.correct} / ${result.total} checks passed</strong><p>${result.issues.length} items need attention. Click an explanation to find its slot.</p><ol>${result.issues.slice(0, 5).map(i => `<li><button data-action="locate" data-issue="${escape(i.id)}" data-module="${escape(i.module || '')}">${escape(i.message)}</button></li>`).join('')}</ol>${result.issues.length > 5 ? `<p>${result.issues.length - 5} more items remain after these.</p>` : ''}</div>`;
  }
  function openDialog(type, id = null) {
    if (!dialog) returnFocus = document.activeElement;
    dialog = { type, id }; activeTarget = null; maskQuery = 2; moduleFeedback = null;
    if (['module', 'residual', 'operation'].includes(type)) selected = null;
    renderDialog(true);
  }
  function closeDialog() {
    dialog = null; activeTarget = null; renderedDialogKey = null; $('#overlay-root').innerHTML = '';
    document.body.classList.remove('modal-open'); $('#app').inert = false; updateSelection();
    if (returnFocus?.isConnected) returnFocus.focus(); else $('.primary-button')?.focus();
  }
  function dialogFrame(title, eyebrow, content, className = '') {
    return `<div class="modal-backdrop"><section class="dialog ${className}" role="dialog" aria-modal="true" aria-labelledby="dialog-title" tabindex="-1"><header class="dialog-header"><div>${eyebrow ? `<span class="overline">${eyebrow}</span>` : ''}<h2 id="dialog-title">${title}</h2></div><button class="icon-button close-dialog" data-action="close" aria-label="Close">${icons.close}</button></header>${content}</section></div>`;
  }
  function renderDialog(focus) {
    if (!dialog) return;
    const key = dialog.type + ':' + (dialog.id || '');
    const scroll = renderedDialogKey === key ? captureScroll(['.workbench-main', '.workbench-palette', '.theory-content', '.guide-content', '.paper-content', '.settings-content']) : [], focused = focusIdentity();
    const templates = { module: () => moduleDialogHtml(dialog.id), residual: () => residualDialogHtml(dialog.id), operation: () => operationDialogHtml(dialog.id), theory: theoryHtml, reference: () => referenceHtml(dialog.id || game.level), figure: () => figureHtml(dialog.id), guide: guideHtml, settings: settingsHtml, success: successHtml };
    let html;
    if (templates[dialog.type]) html = templates[dialog.type]();
    else if (dialog.type === 'reset') html = dialogFrame('Reset this build?', '', '<div class="simple-dialog-content"><p>All blocks in this level will be removed. Completed levels are saved, and you can undo the reset.</p><div class="dialog-actions"><button class="secondary-button" data-action="close">Keep building</button><button class="primary-button" data-action="confirm-reset">Reset build</button></div></div>', 'small-dialog');
    $('#overlay-root').innerHTML = html; document.body.classList.add('modal-open'); $('#app').inert = true;
    if (focus) $('.dialog').focus(); else restoreFocus(focused);
    restoreScroll(scroll); drawResiduals(); renderedDialogKey = key;
  }
  function settingsHtml() {
    const p = PROFILES[game.level];
    return dialogFrame('Build settings', p.label, `<div class="settings-content"><div class="settings-grid"><section><h3>Normalization order</h3><div class="segmented" role="group" aria-label="Normalization order">${['post', 'pre'].map(n => `<button data-action="norm" data-norm="${n}" class="${game.norm === n ? 'active' : ''}" aria-pressed="${game.norm === n}">${PROFILES[game.level].normType === 'rmsnorm' ? (n === 'post' ? 'Post-RMSNorm' : 'Pre-RMSNorm') : n === 'post' ? 'Post-LN' : 'Pre-LN'}</button>`).join('')}</div><code>${game.norm === 'pre' ? 'x + F(' + (PROFILES[game.level].normType === 'rmsnorm' ? 'RMSNorm' : 'LayerNorm') + '(x))' : (PROFILES[game.level].normType === 'rmsnorm' ? 'RMSNorm' : 'LayerNorm') + '(x + F(x))'}</code><p class="norm-origin">Original: ${PAPERS[game.level].norm}</p>${game.norm !== p.norm ? '<button class="text-button" data-action="restore-norm">Restore original order</button>' : ''}<button class="text-button" data-action="theory">Explain Pre-LN vs Post-LN →</button><label class="experiment-toggle"><input type="checkbox" id="experimental" ${game.experimental ? 'checked' : ''}><span><strong>Experiment mode</strong><span>${p.gatedFFN ? 'Try different normalization, widths, or attention head counts.' : 'Try a different order, RMSNorm, other activations, or a deeper FFN.'}</span></span></label><p class="build-note">Experiments are checked for valid shapes and structure; they do not award original model completion.</p></section><section class="model-config"><h3>Dimensions</h3><label><span>Model width d_model</span><select data-config="d" aria-label="Model width">${[64, 128, 256, 512, 768, 1024, 2304, 5120].map(n => `<option ${game.d === n ? 'selected' : ''}>${n}</option>`).join('')}</select></label><label><span>Attention heads</span><select data-config="heads" aria-label="Attention heads">${[1, 2, 4, 8, 12, 16, 24, 32].map(n => `<option ${game.heads === n ? 'selected' : ''}>${n}</option>`).join('')}</select></label><label><span>${p.experts ? 'Expert hidden width' : 'FFN hidden width'}</span><input type="number" data-config="hidden" value="${game.hidden}" min="1" max="32768" aria-label="FFN hidden width"></label>${p.kvHeads ? `<p class="head-dimension">Attention: ${game.heads} Q heads · ${p.kvHeads} K/V heads · d_head = ${p.headDim}.${game.level === 'qwen' ? '<br>DeltaNet: 16 Q/K heads · 48 V heads · d_head = 128.' : `<br>${p.experts} experts · ${p.topK} active per token · 896 features per expert.`}</p>` : `<p class="head-dimension ${Number.isInteger(game.d / game.heads) ? '' : 'dimension-error'}">d_head = ${game.d} / ${game.heads} = <strong>${Number.isInteger(game.d / game.heads) ? game.d / game.heads : 'not an integer'}</strong></p>`}</section></div><div class="settings-actions"><button class="secondary-button" data-action="projector" aria-pressed="${projector}">Text size: ${projector ? 'Large' : 'Standard'}</button><button class="secondary-button" data-action="show-solution">Show correct solution</button><button class="secondary-button" data-action="export">Export JSON</button><button class="secondary-button danger-button" data-action="reset">Reset build</button></div></div><footer class="dialog-footer"><span>Changes save automatically.</span><button class="primary-button" data-action="close">Back to build</button></footer>`, 'settings-dialog');
  }
  function moduleDialogHtml(id) {
    const lane = id.split('.')[0], meta = E.modulesFor(game, lane).find(m => m.id === id), m = E.ensureModule(game, id), ownIssues = E.check(game).issues.filter(i => i.module === id);
    const status = moduleFeedback ? `<div class="module-validation ${ownIssues.length ? 'warning' : 'success'}">${ownIssues.length ? ownIssues.slice(0, 4).map(i => `<p>${escape(i.message)}</p>`).join('') : `<p>✓ Module is correct. ${meta.kind === 'attention' ? 'Q/K/V sources and mask checked.' : 'Output is compatible with the residual.'}</p>`}</div>` : '';
    const palette = `<aside class="workbench-palette"><h3>Building blocks</h3><p>Choose a block and a slot.</p><div class="mini-block-list">${BLOCKS.filter(b => !['ffn', 'moe'].includes(meta.kind) ? ['attention', 'norm', 'ffn'].includes(b.group) || (PROFILES[game.level].kvHeads && ['rope', 'rope-yarn'].includes(b.id)) : ['ffn', 'norm'].includes(b.group) || b.id === 'linear' || (meta.kind === 'moe' && b.id === 'softmax')).map(libraryBlock).join('')}</div><div class="selection-info" aria-live="polite">${selectionHtml()}</div></aside>`;
    return dialogFrame(meta.name, `${lane} block / individual operations`, `<div class="workbench-layout">${palette}<div class="workbench-main">${meta.kind === 'moe' ? moeHtml(meta, m) : meta.kind === 'delta' ? deltaHtml(meta, m) : meta.kind === 'attention' ? attentionHtml(meta, m) : m.gated ? swiGLUHtml(meta, m) : ffnHtml(meta, m)}${status}</div></div><footer class="dialog-footer"><span class="selected-block-label">${selected ? 'Selected: ' + BY_ID[selected].name : 'You can choose a slot first, too.'}</span><div><button class="secondary-button" data-action="check-module">Check module</button><button class="primary-button" data-action="close">Back to build</button></div></footer>`, 'workbench-dialog');
  }

  function operationDialogHtml(id) {
    const coarse = E.COARSE_BY_ID[E.ensureStructure(game)[id]], issues = E.check(game).issues.filter(i => i.id === id);
    const status = moduleFeedback ? `<div class="module-validation ${issues.length ? 'warning' : 'success'}">${issues.length ? issues.map(i => `<p>${escape(i.message)}</p>`).join('') : '<p>✓ Operation is correct.</p>'}</div>` : '';
    const palette = `<aside class="workbench-palette"><h3>Building blocks</h3><div class="mini-block-list">${BLOCKS.filter(b => ['norm', 'ffn'].includes(b.group)).map(libraryBlock).join('')}</div><div class="selection-info">${selectionHtml()}</div></aside>`;
    return dialogFrame(coarse.name, 'Build the operation inside', `<div class="workbench-layout">${palette}<div class="workbench-main"><div class="single-operation">${slotHtml(id, { bare: true })}</div>${status}</div></div><footer class="dialog-footer"><span class="selected-block-label">Choose an operation, then a slot.</span><div><button class="secondary-button" data-action="check-operation">Check operation</button><button class="primary-button" data-action="close">Back to build</button></div></footer>`, 'workbench-dialog');
  }

  function residualDialogHtml(id) {
    const lane = id.split('.')[0], meta = E.modulesFor(game, lane).find(m => m.id === id);
    const issues = E.check(game).issues.filter(i => i.id === id + '.norm' || i.id === id + '.add');
    const status = moduleFeedback ? `<div class="module-validation ${issues.length ? 'warning' : 'success'}">${issues.length ? issues.map(i => `<p>${escape(i.message)}</p>`).join('') : '<p>✓ Residual addition and normalization are correct.</p>'}</div>` : '';
    const palette = `<aside class="workbench-palette"><h3>Building blocks</h3><p>Place Add and normalization separately.</p>${BLOCKS.filter(b => b.group === 'norm' || b.group === 'ffn').map(libraryBlock).join('')}<div class="selection-info">${selectionHtml()}</div></aside>`;
    return dialogFrame(game.norm === 'pre' ? 'Pre-LN Residual' : 'Add & Norm', meta.name + ' · ' + lane, `<div class="workbench-layout">${palette}<div class="workbench-main"><div class="workbench-intro"><p>${game.norm === 'pre' ? 'Normalize before the sublayer. Add the original x afterward.' : 'Add the original x after the sublayer, then normalize.'}</p><code>${game.norm === 'pre' ? 'x + F(Norm(x))' : 'Norm(x + F(x))'}</code></div><div class="residual-builder">${sublayerHtml(meta, 0)}</div>${status}</div></div><footer class="dialog-footer"><span class="selected-block-label">Choose an operation, then a slot.</span><div><button class="secondary-button" data-action="check-residual">Check wrapper</button><button class="primary-button" data-action="close">Back to build</button></div></footer>`, 'workbench-dialog');
  }
  function attentionHtml(meta, m) {
    if (PROFILES[game.level].kvHeads) return groupedAttentionHtml(meta, m);
    const slot = (key, caption) => slotHtml(meta.id + '.' + key, { target: meta.id + '::' + key, value: m.slots[key], label: ATTENTION_STEPS.find(s => s.key === key)?.label, caption }), cross = meta.id.endsWith('.cross');
    return `<div class="workbench-intro"><p>${cross ? 'Q comes from the decoder. K and V come from encoder memory.' : 'Query, Key, and Value are three separate projections of the input.'}</p><span class="dimension-pill">${game.heads} heads · d_head = ${Number.isInteger(game.d / game.heads) ? game.d / game.heads : '?'}</span></div><div class="attention-graph"><section class="attention-section"><h3>1. Create Q, K, V</h3><div class="qkv-row">${['q', 'k', 'v'].map(role => `<div class="projection"><label><span class="projection-letter">${role.toUpperCase()} source</span><select data-source="${role}" data-module="${meta.id}" aria-label="${role.toUpperCase()} source"><option value="hidden" ${m.sources[role] === 'hidden' ? 'selected' : ''}>${cross ? 'Decoder states' : 'Current input'}</option><option value="encoder" ${m.sources[role] === 'encoder' ? 'selected' : ''}>Encoder output</option></select></label>${wire()}${slot(role, `W_${role}: ${game.d} → ${game.d}`)}</div>`).join('')}</div>${wire()}<div class="single-operation">${slot('split', '[B, h, T, d_head]')}</div></section><section class="attention-section"><h3>2. Compute attention weights</h3><div class="operations-grid four">${[['scores', cross ? 'T_decoder × T_encoder' : 'T × T'], ['scale', 'scores / √d_head'], ['mask', 'Before softmax'], ['softmax', 'Normalize over keys']].map(([key, caption]) => slot(key, caption)).join('')}</div><p class="flow-instruction">Run left to right →</p></section><section class="attention-section"><h3>3. Combine values and heads</h3><div class="operations-grid three">${[['weighted', 'weights × V'], ['concat', 'h · d_head = d_model'], ['out', `W_O: ${game.d} → ${game.d}`]].map(([key, caption]) => slot(key, caption)).join('')}</div><p class="flow-instruction">Run left to right →</p></section></div><div class="attention-bottom"><div class="attention-equation"><h3>Inside each head</h3><code>softmax(QKᵀ / √dₖ + mask) V</code><p>After merging the heads, apply the learned output projection W_O.</p><p class="build-note">Dropout and padding masks are omitted in this teaching diagram.</p></div>${maskHtml(meta, m)}</div>`;
  }
  function graphSlot(meta, m, key, caption = '') {
    return slotHtml(meta.id + '.' + key, { target: meta.id + '::' + key, value: m.slots[key], label: E.moduleSteps(game, meta).find(s => s.key === key)?.label, caption });
  }
  function operationChain(meta, m, items) { return items.map(item => graphSlot(meta, m, ...(Array.isArray(item) ? item : [item]))).join(wire()); }
  function groupedAttentionHtml(meta, m) {
    const p = PROFILES[game.level], hd = E.headDim(game), qWidth = game.heads * hd, kvWidth = p.kvHeads * hd;
    const slot = (key, caption) => graphSlot(meta, m, key, caption);
    return `<div class="workbench-intro"><p>Grouped-query attention: ${game.heads} query heads share ${p.kvHeads} K/V heads. Head width is ${hd}; it is independent of d_model.</p><span class="dimension-pill">${game.heads} Q · ${p.kvHeads} K/V · ${game.level === 'qwen' ? 64 : hd} rotary features</span></div>
      <div class="attention-graph"><section class="attention-section"><h3>1. Project and position the heads</h3><div class="qkv-row">${['q','k','v'].map(role => `<div class="projection"><label>${role.toUpperCase()} source<select data-source="${role}" data-module="${meta.id}"><option value="hidden" ${m.sources[role] === 'hidden' ? 'selected' : ''}>Current input</option><option value="encoder" ${m.sources[role] === 'encoder' ? 'selected' : ''}>Encoder output</option></select></label>${slot(role, role === 'q' ? `${game.d} → ${qWidth}` : `${game.d} → ${kvWidth}`)}</div>`).join('')}</div>${wire()}<div class="single-operation">${slot('split', `Q: [B, ${game.heads}, T, ${hd}]; K/V: [B, ${p.kvHeads}, T, ${hd}]`)}</div><div class="operations-grid two">${slot('qNorm', 'Q only · per head')}${slot('kNorm', 'K only · per head')}</div>${wire()}<div class="operations-grid two">${slot('rope', game.level === 'qwen' ? 'Q/K: rotate 64 of 256 features; V unchanged' : meta.rope === 'rope-yarn' ? 'Q/K: YaRN factor 16, reference context 8192; V unchanged' : 'Q/K: RoPE across 128 features; V unchanged')}${slot('repeat', `Each K/V head serves ${game.heads / p.kvHeads} Q heads`)}</div></section>
      <section class="attention-section"><h3>2. Causal attention</h3><div class="operations-grid four">${[['scores','T × T'],['scale',`scores / √${hd}`],['mask',meta.mask === 'sliding-causal' ? `Causal window = ${p.window}` : 'Hide future tokens'],['softmax','Normalize over keys']].map(([k,c]) => slot(k,c)).join('')}</div><p class="flow-instruction">Run left to right →</p><div class="operations-grid two">${slot('weighted','weights × V')}${slot('concat',`${game.heads} × ${hd} = ${qWidth}`)}</div></section>
      ${p.outputGate ? `<section class="attention-section"><h3>3. Gate the output</h3><div class="gate-branches"><div class="branch-card"><h4>From current input</h4>${operationChain(meta,m,[['gateProj',`${game.d} → ${qWidth}`],['gateAct','Output gate: SiLU (swish)']])}</div><div class="branch-card"><h4>From attention output</h4><p>Concatenated head states</p><p>[B, T, ${qWidth}]</p></div></div><div class="branch-merge">↘ &nbsp; ↙</div><div class="single-operation">${slot('gate','Attention output × SiLU(gate)')}${wire()}${slot('out',`${qWidth} → ${game.d}`)}</div></section>` : `<section class="attention-section"><h3>3. Project to model width</h3><div class="single-operation">${slot('out',`${qWidth} → ${game.d}`)}</div></section>`}</div><p class="build-note">${p.outputGate ? 'Q and gate projections are shown separately for clarity; the implementation fuses them in q_proj. This checkpoint specifies output_gate_type = swish in its configuration.' : 'Attention output goes directly through W_O. Mellum2 uses per-head Q/K RMSNorm and has no attention output gate.'}</p>${maskHtml(meta,m)}`;
  }
  function deltaHtml(meta, m) {
    const slot = (key, caption) => graphSlot(meta, m, key, caption);
    return `<div class="workbench-intro"><p>A causal matrix memory replaces the T × T softmax attention matrix. This layer does not use RoPE.</p><span class="dimension-pill">16 Q/K heads · 48 V heads · 128 features</span></div>
      <section class="attention-section"><h3>1. Prepare Q, K, V</h3><div class="delta-preparation">${operationChain(meta,m,[['qkv',`${game.d} → 2048 + 2048 + 6144`],['conv','Depthwise kernel = 4 · causal'],['convAct','Applied to convolved Q/K/V'],['split','Q/K: 16 × 128; V: 48 × 128'],['qkNorm','Q/K only, not V'],['repeat','Each Q/K head serves 3 V heads'],['scale','Q / √128']])}</div></section>
      <section class="attention-section"><h3>2. Control and update memory</h3><div class="gate-branches"><div class="branch-card"><h4>Update strength · from input</h4>${operationChain(meta,m,[['betaProj',`${game.d} → 48`],['beta','beta = sigmoid(b)']])}</div><div class="branch-card"><h4>State decay · from input</h4>${operationChain(meta,m,[['decayProj',`${game.d} → 48`],['timeStep','softplus(a + dt_bias)'],['decay','alpha = exp(−exp(A_log) × time_step)']])}</div></div><div class="branch-merge">↘ &nbsp; ↙</div><div class="single-operation">${slot('rule','Decay S → delta correction → read with scaled Q')}</div><pre class="delta-equation">S̄ = alpha · S_previous
S = S̄ + beta · k (v − kᵀS̄)ᵀ
o = qᵀS</pre><p class="build-note">The state is [B, 48, 128, 128]. S starts at zero and updates in token order. The block above combines the recurrent update and read; chunked training computes the same causal operation.</p></section>
      <section class="attention-section"><h3>3. Normalize, gate, and project</h3><div class="gate-branches"><div class="branch-card"><h4>From memory output</h4>${slot('outNorm','RMSNorm separately per 128-feature value head')}</div><div class="branch-card"><h4>From current input</h4>${operationChain(meta,m,[['gateProj',`${game.d} → 6144`],['gateAct','SiLU(z)']])}</div></div><div class="branch-merge">↘ &nbsp; ↙</div><div class="single-operation">${operationChain(meta,m,[['gate','RMSNorm(o) × SiLU(z)'],['concat','48 × 128 = 6144'],['out',`6144 → ${game.d}`]])}</div></section>`;
  }
  function swiGLUHtml(meta, m) {
    const width = key => `<label class="branch-width">Output width <input type="number" min="1" max="32768" data-branch-width="${key}" data-module="${meta.id}" value="${m.widths[key]}" aria-label="${key} output width"></label>`;
    return `<div class="workbench-intro"><p>Two projections take the same input. Activate the gate branch, multiply both branches, then project down.</p><span class="dimension-pill">${game.d} → ${game.hidden} → ${game.d}</span></div><div class="gate-branches"><div class="branch-card"><h3>Gate branch</h3>${graphSlot(meta,m,'gateProj',`Input: ${game.d}`)}${width('gateProj')}${wire()}${graphSlot(meta,m,'activation','SiLU(gate_proj(x))')}</div><div class="branch-card"><h3>Value branch</h3>${graphSlot(meta,m,'upProj',`Same input: ${game.d}`)}${width('upProj')}</div></div><div class="branch-merge">↘ &nbsp; ↙</div><div class="single-operation">${graphSlot(meta,m,'multiply','Both branches must have the same width')}${wire()}${graphSlot(meta,m,'downProj',`Return to d_model = ${game.d}`)}${width('downProj')}</div><div class="ffn-info"><strong>SwiGLU</strong><code>down_proj(SiLU(gate_proj(x)) × up_proj(x))</code><p>A single SiLU layer is only the activation. SwiGLU also needs the parallel value branch and elementwise multiplication.</p></div>`;
  }
  function moeHtml(meta, m) {
    const p = PROFILES[game.level], slot = (key, caption) => graphSlot(meta, m, key, caption);
    const routingChoice = (key, label, values) => `<label class="branch-width">${label}<select data-routing="${key}" data-module="${meta.id}" aria-label="${label}">${values.map(n => `<option value="${n}" ${m.routing[key] === n ? 'selected' : ''}>${n}</option>`).join('')}</select></label>`;
    return `<div class="workbench-intro"><p>A router chooses a different set of experts for each token. The selected experts run in parallel on the same input.</p><span class="dimension-pill">${p.experts} experts · top ${p.topK} per token</span></div>
      <section class="attention-section"><h3>1. Route each token</h3>${routingChoice('experts','Number of experts',[16,32,64,128])}${routingChoice('topK','Experts per token',[1,2,4,8,16])}<div class="operations-grid three">${[['routerProj',`${game.d} → ${m.routing.experts}`],['routerSoftmax','Across all experts'],['topK',`Select ${m.routing.topK} experts`]].map(([k,c]) => slot(k,c)).join('')}</div><p class="flow-instruction">Run left to right →</p><div class="operations-grid two">${slot('renormalize','Selected weights sum to 1')}${slot('dispatch','Each chosen expert receives the token input')}</div></section>
      <section class="attention-section"><h3>2. Build one SwiGLU expert</h3><p class="build-note">Repeat this structure for all ${p.experts} experts with independent weights. Only ${p.topK} run for a given token. Each expert’s hidden width is ${game.hidden}, rather than the dense configuration width ${p.denseHidden}.</p>${swiGLUHtml(meta,m)}</section>
      <section class="attention-section"><h3>3. Return one vector per token</h3><div class="single-operation">${slot('combine',`Sum of ${p.topK} expert outputs × routing weights → ${game.d}`)}</div><pre class="delta-equation">y = Σ selected_experts routing_weight · expert(x)</pre></section>`;
  }
  function maskHtml(meta, m) {
    const sliding = m.slots.mask === 'sliding-causal', causal = m.slots.mask === 'causal' || sliding, toyWindow = 3, none = !m.slots.mask, cross = meta.id.endsWith('.cross');
    const rows = cross ? PROFILES.encdec.tokens : PROFILES[game.level].tokens, cols = cross ? ['the', 'cat', 'sits', 'on', 'mat', '<EOS>'] : rows;
    return `<div class="mask-demo"><h3>Visible context <span>${none ? 'No mask yet' : sliding ? 'sliding causal' : causal ? 'causal' : 'full'}</span></h3><div class="mask-grid"><div></div>${cols.map((t, j) => `<span class="matrix-col" title="${escape(t)}">${j + 1}</span>`).join('')}${rows.map((t, i) => `<button data-action="mask-query" data-query="${i}" class="matrix-query ${maskQuery === i ? 'active' : ''}" title="${escape(t)}" aria-label="Show context for ${escape(t)}">${i + 1}</button>${cols.map((_, j) => `<span class="matrix-cell ${none ? 'unset' : !causal || (j <= i && (!sliding || j > i - toyWindow)) ? 'allowed' : 'blocked'} ${maskQuery === i ? 'query-row' : ''}" title="${escape(t)} → ${escape(cols[j])}: ${none ? 'mask not chosen' : !causal || (j <= i && (!sliding || j > i - toyWindow)) ? 'visible' : 'hidden'}"></span>`).join('')}`).join('')}</div><p>${none ? 'This shows access, not softmax weights.' : `Query “${escape(rows[maskQuery])}” sees ${sliding ? Math.min(maskQuery + 1, toyWindow) : causal ? maskQuery + 1 : 6} of 6 tokens.`}${sliding ? ` Toy window: ${toyWindow} tokens; actual model window: ${PROFILES[game.level].window}.` : ''}</p></div>`;
  }
  function ffnHtml(meta, m) {
    let currentWidth = game.d;
    const rows = m.layers.map((layer, i) => {
      const inputWidth = currentWidth; if (layer.type === 'linear') currentWidth = Number(layer.width);
      return `<div class="ffn-layer-row" draggable="true" data-layer-index="${i}"><span class="layer-index">${i + 1}</span><div class="ffn-operation">${slotHtml(meta.id + '.layer.' + i, { target: meta.id + '::layer:' + i, value: layer.type, label: 'FFN operation ' + (i + 1) })}</div><div class="ffn-shape"><span>${inputWidth} →</span>${layer.type === 'linear' ? `<input type="number" value="${layer.width}" min="1" max="32768" data-width="${i}" data-module="${meta.id}" aria-label="Linear ${i + 1} output width">` : `<strong>${inputWidth}</strong>`}</div><div class="ffn-order"><button data-action="move-layer" data-index="${i}" data-direction="-1" aria-label="Move operation ${i + 1} up" ${i === 0 ? 'disabled' : ''}>↑</button><button data-action="move-layer" data-index="${i}" data-direction="1" aria-label="Move operation ${i + 1} down" ${i === m.layers.length - 1 ? 'disabled' : ''}>↓</button><button data-action="delete-layer" data-index="${i}" aria-label="Delete operation ${i + 1}">×</button></div></div>`;
    }).join('<div class="ffn-flow-arrow">↓</div>');
    return `<div class="workbench-intro"><p>The same network runs independently on every token.</p><span class="dimension-pill">Input: [B, T, ${game.d}]</span></div><div class="ffn-builder"><div class="ffn-boundary">Input · d_model = ${game.d}</div>${rows}<button class="add-layer-button" data-action="add-layer">+ Add operation</button><div class="ffn-boundary ${currentWidth !== game.d ? 'shape-mismatch' : ''}">Output: ${currentWidth} ${currentWidth !== game.d ? '≠' : '='} d_model · ${currentWidth === game.d ? 'residual compatible' : 'residual shape mismatch'}</div></div><div class="ffn-info"><strong>${game.experimental ? 'Experimental FFN' : 'Reference: Linear → ' + BY_ID[PROFILES[game.level].activation].name + ' → Linear'}</strong><p>${game.experimental ? 'Add operations and activations. Input and output must have d_model features, with at least one nonlinearity.' : `Expand ${game.d} → ${game.hidden}, then return to ${game.d}. Enable experiment mode for a different activation or depth.`}</p><button class="text-button" data-action="toggle-experiment">${game.experimental ? 'Use original model checks' : 'Enable experiment mode'}</button></div>`;
  }
  function paperImage(fig) { return window.TransformerPaperImages?.[fig.id] || fig.file; }
  function referenceHtml(level) {
    const paper = PAPERS[level];
    return dialogFrame('Paper & original figures', paper.model, `<nav class="paper-tabs" aria-label="Model papers">${Object.keys(PROFILES).map(key => `<button data-action="paper" data-paper="${key}" class="${key === level ? 'active' : ''}" aria-pressed="${key === level}">${PROFILES[key].name} <span class="model-year">${PROFILES[key].year}</span><span>${PAPERS[key].norm}</span></button>`).join('')}</nav><div class="paper-content"><div class="paper-overview"><div><p>${paper.authors} · ${paper.year}</p><h3>${paper.title}</h3><div class="paper-links"><a href="${paper.url}" target="_blank" rel="noopener">${paper.pdf ? 'Read paper' : 'Official model card'} ↗</a>${paper.pdf ? `<a href="${paper.pdf}" target="_blank" rel="noopener">Original PDF ↗</a>` : ''}${paper.config ? `<a href="${paper.config}" target="_blank" rel="noopener">Checkpoint config ↗</a>` : ''}${paper.code ? `<a href="${paper.code}" target="_blank" rel="noopener">Original code ↗</a>` : ''}</div></div><span class="paper-norm-badge">${paper.norm}</span></div><div class="paper-architecture"><div><h3>Reference architecture</h3><ul>${paper.facts.map(f => `<li>${f}</li>`).join('')}</ul><p>${paper.normSummary}</p></div><pre>${escape(paper.formula)}</pre></div><div class="paper-evidence"><strong>Where normalization order is specified</strong><p>${paper.normEvidence}</p>${paper.quote ? `<blockquote>${paper.quote}</blockquote>` : ''}</div>${paper.note ? `<p class="paper-note">${paper.note}</p>` : ''}${paper.documents ? `<div class="paper-documents"><h3>Offline source documents</h3>${paper.documents.map(doc => `<a href="${doc.file}" download>${doc.title} ↓</a>`).join('')}</div>` : ''}<div class="paper-figures">${paper.figures.map((fig, index) => `<figure class="paper-figure"><div class="figure-heading"><h4>${fig.title}</h4><a href="${paper.pdf}#page=${fig.page}" target="_blank" rel="noopener">Page ${fig.page} ↗</a></div><button class="figure-image-button" data-action="figure" data-figure="${level}.${index}" aria-label="Enlarge ${escape(fig.title)}"><img src="${paperImage(fig)}" data-paper-figure="${fig.id}" data-fallback="${fig.file}" alt="${escape(fig.title + '. ' + fig.description)}" decoding="async"><span class="figure-zoom">Enlarge ↗</span></button><figcaption>${fig.description}<a href="${fig.file}" download="${fig.id}.png">Download PNG ↓</a></figcaption></figure>`).join('')}</div><p class="paper-offline-note">${paper.figures.length ? 'These images were extracted from the original PDFs and are embedded in the game. They work offline.' : 'The original checkpoint configuration and model-card excerpt are included for offline reference.'}</p></div><footer class="dialog-footer"><button class="text-button" data-action="theory">Pre-LN vs Post-LN</button><div>${level !== game.level ? `<button class="secondary-button" data-action="next-level" data-level="${level}">Open this model</button>` : ''}<button class="primary-button" data-action="close">Back to build</button></div></footer>`, 'paper-dialog');
  }
  function figureHtml(id) {
    const [level, index] = id.split('.'), paper = PAPERS[level], fig = paper.figures[Number(index)];
    return dialogFrame(fig.title, `${paper.model} · original · page ${fig.page}`, `<div class="figure-preview"><img src="${paperImage(fig)}" data-paper-figure="${fig.id}" data-fallback="${fig.file}" alt="${escape(fig.description)}"><p>${fig.description}</p></div><footer class="dialog-footer"><a class="text-button" href="${fig.file}" download="${fig.id}.png">Download PNG</a><div><a class="secondary-button" href="${paper.pdf}#page=${fig.page}" target="_blank" rel="noopener">Open PDF ↗</a><button class="primary-button" data-action="back-paper" data-paper="${level}">Back to paper</button></div></footer>`, 'figure-dialog');
  }
  function theoryHtml() {
    return dialogFrame('Pre-LN or Post-LN?', 'Normalization order', `<div class="theory-content"><p class="theory-lead">The original Transformer uses Post-LN, a valid architecture. Pre-LN is common in deep GPT-style models because the direct residual path often makes optimization more stable.</p><div class="norm-comparison"><article><h3>Post-LN</h3><strong>Original Transformer · BERT</strong><div class="norm-diagram">x → F → Add x → LayerNorm</div><code>y = LayerNorm(x + F(x))</code><p>Normalize after the sublayer and residual addition. Deep stacks can require careful training settings and warmup.</p></article><article><h3>Pre-LN</h3><strong>GPT-2 · many modern LLMs</strong><div class="norm-diagram">x → LayerNorm → F → Add x</div><code>y = x + F(LayerNorm(x))</code><p>Normalize the sublayer input. The original x bypasses normalization and the sublayer. A final normalization usually follows the stack.</p></article></div><div class="theory-callout"><h3>Order and normalization type are separate choices</h3><p>GPT-2 uses <strong>LayerNorm</strong> in Pre-LN order, plus final LayerNorm. Qwen3.8 uses <strong>RMSNorm</strong> in pre-norm order, plus final RMSNorm. Llama uses RMSNorm with pre-norm. T5 uses pre-norm with an RMS-like normalization. Some newer models combine normalization placements.</p></div><div class="profile-table-wrap"><table class="profile-table"><thead><tr><th>Reference</th><th>Attention</th><th>Normalization</th><th>Positions / FFN</th></tr></thead><tbody><tr><td>BERT-base</td><td>Bidirectional self</td><td>Post-LN</td><td>Learned + token type / GELU</td></tr><tr><td>GPT-2 small</td><td>Causal self</td><td>Pre-LN + final LN</td><td>Learned / GELU</td></tr><tr><td>Mellum 2</td><td>Sliding + full causal GQA</td><td>Pre-RMSNorm + final RMSNorm</td><td>RoPE / YaRN + sparse SwiGLU experts</td></tr><tr><td>Qwen3.8-27B</td><td>Gated DeltaNet + causal GQA</td><td>Pre-RMSNorm + final RMSNorm</td><td>Partial RoPE in attention / SwiGLU</td></tr><tr><td>Transformer 2017</td><td>Encoder self + decoder causal + cross</td><td>Post-LN</td><td>Sinusoidal / ReLU</td></tr></tbody></table></div><p class="build-note">The game shows the inference structure, omitting dropout, padding masks, embedding scaling √d_model, and BERT pre-training heads. RoPE rotates Q/K inside attention. SwiGLU needs a gated branch; a single SiLU operation is not SwiGLU.</p><div class="theory-sources"><a href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noopener">Transformer paper ↗</a><a href="https://arxiv.org/abs/1810.04805" target="_blank" rel="noopener">BERT paper ↗</a><a href="${PAPERS.gpt.pdf}" target="_blank" rel="noopener">GPT-2 paper ↗</a><a href="${PAPERS.gpt.code}" target="_blank" rel="noopener">GPT-2 code ↗</a><a href="https://arxiv.org/abs/2002.04745" target="_blank" rel="noopener">On Layer Normalization ↗</a><a href="https://arxiv.org/abs/1910.07467" target="_blank" rel="noopener">RMSNorm ↗</a><a href="https://arxiv.org/abs/2302.13971" target="_blank" rel="noopener">Llama ↗</a></div></div>`, 'theory-dialog');
  }
  function guideHtml() {
    return dialogFrame('How to play', '', `<div class="guide-content"><p>Build the full Architecture on one canvas. Every empty block is a choice; the main flow runs from bottom to top. Side paths show residual connections, and encoder memory feeds decoder cross-attention.</p><p>Choose a module and click a slot, choose the slot first, or drag and drop. Once the architecture is correct, switch to Inside blocks or use ↗ to build a module’s operations. Check build verifies the whole model.</p><p>Show correct solution fills all modules and operations immediately, restoring the original model settings. Undo brings back your previous build. Settings contains dimensions, normalization order, experiment mode, text size, export, and reset. Existing internal work stays saved when you change the architecture.</p><div class="keyboard-help"><span><kbd>Esc</kbd> close / deselect</span><span><kbd>⌘ / Ctrl + Z</kbd> undo</span><span><kbd>/</kbd> search</span></div><div class="guide-footer"><button class="secondary-button" data-action="show-solution">Show correct solution</button><button class="primary-button" data-action="close">Back to build</button></div></div>`, 'guide-dialog');
  }

  function successHtml() {
    const result = E.check(game), keys = Object.keys(PROFILES), next = keys[keys.indexOf(game.level) + 1];
    return dialogFrame(`${game.experimental ? 'Experiment' : PROFILES[game.level].name} complete!`, '', `<div class="success-content"><h3>All ${result.correct} checks passed.</h3><p>Operations, Q/K/V sources, masks, and dimensions are consistent.</p>${result.notes.map(n => `<p>${escape(n)}</p>`).join('')}<div class="success-stats"><div><strong>${game.heads}</strong><span>attention heads</span></div><div><strong>${E.headDim(game)}</strong><span>features per attention head</span></div><div><strong>${E.progress(game).filled}</strong><span>operations placed</span></div></div><div class="dialog-actions"><button class="secondary-button" data-action="success-trace">Trace shapes</button>${next ? `<button class="primary-button" data-action="next-level" data-level="${next}">Next model →</button>` : '<button class="primary-button" data-action="close">Back to build</button>'}</div></div>`, 'success-dialog');
  }
  function applyBlock(target, block) {
    if (!BY_ID[block]) return;
    if (target.startsWith('structure::')) {
      if (!E.COARSE_BY_ID[block]) return toast('Choose a whole module for this architecture slot.');
      const id = target.slice('structure::'.length);
      if (!E.coarseTopology(game).some(s => s.id === id)) return;
      remember(); activeTarget = null; E.ensureStructure(game)[id] = block; persist(); render(); return;
    }
    if (!E.BY_ID[block]) return toast('Choose an individual operation to build inside this module.');
    if (target.includes('::')) { const [id, key] = target.split('::'), m = E.ensureModule(game, id); if (key.startsWith('layer:') && !m.layers[Number(key.split(':')[1])]) return; }
    remember(); activeTarget = null;
    if (target.includes('::')) {
      const [id, key] = target.split('::'), m = E.ensureModule(game, id);
      if (key.startsWith('layer:')) { const index = Number(key.split(':')[1]); m.layers[index].type = block; if (block === 'linear') m.layers[index].width = index === m.layers.length - 1 ? game.d : game.hidden; }
      else m.slots[key] = block;
    } else game.slots[target] = block;
    persist(); render();
  }
  function removeBlock(target) {
    remember(); activeTarget = null;
    if (target.startsWith('structure::')) { delete E.ensureStructure(game)[target.slice('structure::'.length)]; persist(); render(); return; }
    if (target.includes('::')) { const [id, key] = target.split('::'), m = E.ensureModule(game, id); if (key.startsWith('layer:')) m.layers[Number(key.split(':')[1])].type = null; else delete m.slots[key]; }
    else delete game.slots[target]; persist(); render();
  }
  function chooseBlock(id) {
    if (activeTarget && id) { selected = id; applyBlock(activeTarget, id); return; }
    selected = selected === id ? null : id; updateSelection();
  }
  function updateSelection() {
    document.querySelectorAll('[data-action="select"]').forEach(el => { el.classList.toggle('selected', el.dataset.block === selected); el.setAttribute('aria-pressed', String(el.dataset.block === selected)); });
    document.querySelectorAll('.drop-slot').forEach(el => { el.classList.toggle('ready', !!selected); el.classList.toggle('slot-selected', el.dataset.target === activeTarget); });
    document.querySelectorAll('.selection-info').forEach(el => { el.innerHTML = selectionHtml(); });
    if ($('.selected-block-label')) $('.selected-block-label').textContent = activeTarget ? 'Slot selected — choose a block.' : selected ? 'Selected: ' + BY_ID[selected].name : 'You can choose a slot first, too.';
  }
  function switchLevel(level) {
    stopTrace(); if (dialog) closeDialog(); session.active = level; game = session.games[level]; assembly = game.assemblyStage || 'architecture'; feedback = null; selected = null; activeTarget = null; phase = 'input'; persist(); render();
  }
  function showCorrectSolution() {
    remember();
    const reference = E.fillReference(E.createGame(game.level));
    reference.attempts = game.attempts; reference.hints = game.hints; reference.assemblyStage = assembly;
    session.games[game.level] = game = reference;
    selected = null; activeTarget = null; search = ''; filter = 'all';
    if (dialog) closeDialog();
    persist(); render();
    toast('Correct solution filled. Open any module to inspect it. Undo restores your build.');
  }
  function validate() {
    stopTrace(); game.attempts++; feedback = assembly === 'architecture' ? E.checkStructure(game) : E.check(game);
    if (assembly === 'architecture') { persist(); render(); $('.feedback')?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); return; }
    if (feedback.valid) { game.completed = true; if (!game.experimental) session.achievements[game.level] = true; persist(); render(); openDialog('success'); }
    else { persist(); render(); $('.feedback')?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); }
  }
  function hint() {
    const result = assembly === 'architecture' ? E.checkStructure(game) : E.check(game); if (result.valid) return toast(assembly === 'architecture' ? 'Architecture is ready. Continue to Inside blocks.' : 'Everything is in place. Check your build or trace its shapes.');
    game.hints++; persist(); const issue = result.issues[0]; toast(issue.message); locate(issue.id, issue.module);
  }
  function locate(id, module) {
    if (id.startsWith('config.')) { openDialog('settings'); return; }
    assembly = id.startsWith('structure.') ? 'architecture' : 'inside'; game.assemblyStage = assembly; selected = null;
    phase = targetPhase(module || id); activeTarget = null; if (dialog) closeDialog(); persist(); render();
    if (module) { openDialog('module', module); moduleFeedback = E.check(game); renderDialog(false); }
    const el = [...document.querySelectorAll('[data-node]')].find(n => n.dataset.node === id);
    if (el) { el.scrollIntoView({ behavior: 'smooth', block: 'center' }); el.classList.add('hint-highlight'); setTimeout(() => el.classList.remove('hint-highlight'), 2500); }
  }
  function stopTrace() { clearTimeout(traceTimer); traceState = null; document.querySelectorAll('.trace-active').forEach(el => el.classList.remove('trace-active')); $('.trace-bar')?.remove(); }
  function startTrace() {
    if (!E.check(game).valid) { feedback = E.check(game); render(); $('.feedback')?.scrollIntoView({ block: 'nearest' }); toast('Complete a valid build before tracing its shapes.'); return; }
    stopTrace(); assembly = 'inside'; game.assemblyStage = assembly; render(); traceState = { steps: E.trace(game), index: 0 }; showTraceStep();
  }
  function showTraceStep() {
    if (!traceState) return;
    const step = traceState.steps[traceState.index];
    document.querySelectorAll('.trace-active').forEach(el => el.classList.remove('trace-active')); $('.trace-bar')?.remove();
    const node = [...document.querySelectorAll('[data-node]')].find(el => el.dataset.node === step.id); node?.classList.add('trace-active'); node?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    const bar = document.createElement('div'); bar.className = 'trace-bar'; bar.setAttribute('role', 'status');
    bar.innerHTML = `<div><span>Step ${traceState.index + 1} / ${traceState.steps.length}</span><strong>${escape(step.label)}</strong><p>${escape(step.desc)}</p><small>Structure and shape simulation; no model weights are computed.</small></div><button class="icon-button" data-action="stop-trace" aria-label="Stop trace">${icons.close}</button>`; $('.workspace').append(bar);
    traceTimer = setTimeout(() => { if (!traceState) return; if (traceState.index < traceState.steps.length - 1) { traceState.index++; showTraceStep(); } else { stopTrace(); toast(PROFILES[game.level].hybridGroup ? `Trace complete. Repeat the four-layer group × ${PROFILES[game.level].hybridGroup.repeats} for ${game.depth} layers.` : `Trace complete. Each stack block repeats × ${game.depth}.`); } }, 3000);
  }
  function exportGame() {
    const data = { format: 'transformer-architecture', version: 1, profile: PROFILES[game.level].label, game: E.clone(game), validation: E.check(game), route: E.trace(game) }, url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
    const a = document.createElement('a'); a.href = url; a.download = `transformer-${game.level}.json`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); toast('Architecture exported as JSON.');
  }
  async function toggleFullscreen() {
    try { if (document.fullscreenElement) await document.exitFullscreen(); else await document.documentElement.requestFullscreen(); }
    catch (_) { toast('Use your browser’s fullscreen command if fullscreen is unavailable.'); }
  }
  document.addEventListener('fullscreenchange', () => { const button = $('[data-action="fullscreen"]'); if (button) button.textContent = document.fullscreenElement ? 'Exit fullscreen' : 'Fullscreen'; });
  document.addEventListener('click', event => {
    if (event.target.classList.contains('modal-backdrop')) { closeDialog(); return; }
    const button = event.target.closest('[data-action]'); if (!button || button.disabled) return;
    const a = button.dataset.action;
    if (a === 'select') chooseBlock(button.dataset.block);
    else if (a === 'slot') { if (selected) applyBlock(button.dataset.target, selected); else { activeTarget = activeTarget === button.dataset.target ? null : button.dataset.target; updateSelection(); } }
    else if (a === 'remove') removeBlock(button.dataset.target);
    else if (a === 'stage') setAssembly(button.dataset.stage);
    else if (a === 'open-inside') openInside(button.dataset.detail);
    else if (a === 'phase') { stopTrace(); phase = button.dataset.phase; activeTarget = null; render(); }
    else if (a === 'level' || a === 'next-level') switchLevel(button.dataset.level);
    else if (a === 'module') openDialog('module', button.dataset.module);
    else if (a === 'close') closeDialog();
    else if (['theory', 'reference', 'guide', 'settings', 'reset'].includes(a)) openDialog(a);
    else if (a === 'paper' || a === 'back-paper') openDialog('reference', button.dataset.paper);
    else if (a === 'figure') openDialog('figure', button.dataset.figure);
    else if (a === 'projector') { projector = !projector; try { localStorage.setItem('transformer-text-size-v2', projector ? 'large' : 'standard'); } catch (_) {} render(); }
    else if (a === 'fullscreen') toggleFullscreen();
    else if (a === 'restore-norm') { remember(); game.norm = PROFILES[game.level].norm; assembly = 'architecture'; game.assemblyStage = assembly; persist(); render(); toast('Original normalization order restored.'); }
    else if (a === 'filter') { filter = button.dataset.filter; $('#palette-content').innerHTML = paletteHtml(); document.querySelectorAll('[data-action="filter"]').forEach(el => { el.classList.toggle('active', el.dataset.filter === filter); el.setAttribute('aria-pressed', String(el.dataset.filter === filter)); }); }
    else if (a === 'norm' && game.norm !== button.dataset.norm) { remember(); game.norm = button.dataset.norm; assembly = 'architecture'; game.assemblyStage = assembly; persist(); render(); }
    else if (a === 'show-solution') showCorrectSolution();
    else if (a === 'validate') validate();
    else if (a === 'hint') hint();
    else if (a === 'locate') locate(button.dataset.issue, button.dataset.module);
    else if (a === 'undo') undo();
    else if (a === 'confirm-reset') { remember(); const mode = game.norm, experiment = game.experimental, fresh = E.createGame(game.level); fresh.norm = mode; fresh.experimental = experiment; session.games[game.level] = game = fresh; closeDialog(); phase = 'input'; assembly = 'architecture'; persist(); render(); toast('Build reset. You can undo this.'); }
    else if (a === 'check-operation') { moduleFeedback = E.check(game); renderDialog(false); }
    else if (a === 'check-residual') { moduleFeedback = E.check(game); renderDialog(false); drawResiduals(); $('.module-validation')?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); }
    else if (a === 'check-module') { moduleFeedback = E.check(game); renderDialog(false); $('.module-validation')?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); }
    else if (a === 'add-layer') { const m = E.ensureModule(game, dialog.id); if (m.layers.length >= 16) return toast('The FFN supports up to 16 operations.'); remember(); m.layers.push({ type: null, width: game.d }); persist(); render(); }
    else if (a === 'delete-layer') { remember(); E.ensureModule(game, dialog.id).layers.splice(Number(button.dataset.index), 1); persist(); render(); }
    else if (a === 'move-layer') moveLayer(Number(button.dataset.index), Number(button.dataset.index) + Number(button.dataset.direction));
    else if (a === 'mask-query') { maskQuery = Number(button.dataset.query); renderDialog(false); }
    else if (a === 'toggle-experiment') { remember(); game.experimental = !game.experimental; persist(); render(); }
    else if (a === 'trace') startTrace();
    else if (a === 'success-trace') { closeDialog(); startTrace(); }
    else if (a === 'stop-trace') stopTrace();
    else if (a === 'export') exportGame();
  });
  function undo() {
    const h = histories[game.level]; if (!h?.length) return toast('Nothing to undo yet.'); stopTrace(); session.games[game.level] = game = h.pop(); assembly = game.assemblyStage || 'architecture'; feedback = null; moduleFeedback = null; activeTarget = null; persist(); render(); toast('Last action undone.');
  }
  function moveLayer(from, to) { const m = E.ensureModule(game, dialog.id); if (to < 0 || to >= m.layers.length || from === to) return; remember(); const [layer] = m.layers.splice(from, 1); m.layers.splice(to, 0, layer); persist(); render(); }
  document.addEventListener('input', event => { if (event.target.id === 'block-search') { search = event.target.value; $('#palette-content').innerHTML = paletteHtml(); } });
  document.addEventListener('change', event => {
    const el = event.target;
    if (el.dataset.source) { remember(); E.ensureModule(game, el.dataset.module).sources[el.dataset.source] = el.value; persist(); render(); }
    else if (el.dataset.routing) {
      const m = E.ensureModule(game, el.dataset.module), value = Number(el.value);
      if (m.routing[el.dataset.routing] === value) return;
      remember(); m.routing[el.dataset.routing] = value; persist(); render();
    }
    else if (el.dataset.branchWidth) {
      const value = Number(el.value); if (!Number.isInteger(value) || value < 1 || value > 32768) { toast('Use a positive integer up to 32768.'); render(); return; }
      const m = E.ensureModule(game, el.dataset.module); if (m.widths[el.dataset.branchWidth] === value) return;
      remember(); m.widths[el.dataset.branchWidth] = value; persist(); render();
    }
    else if (el.dataset.width !== undefined) { const value = Number(el.value); if (!Number.isInteger(value) || value < 1 || value > 32768) { toast('Use a positive integer up to 32768.'); render(); return; } remember(); E.ensureModule(game, el.dataset.module).layers[Number(el.dataset.width)].width = value; persist(); render(); }
    else if (el.dataset.config) {
      const key = el.dataset.config, value = Number(el.value); if (!Number.isInteger(value) || value < 1 || value > 32768) { toast('Use a positive integer up to 32768.'); render(); return; }
      remember(); const oldD = game.d, oldHidden = game.hidden; game[key] = value; if (key === 'd') game.hidden = value * 4;
      if (key === 'd' || key === 'hidden') for (const m of Object.values(game.modules)) if (m.kind === 'ffn') for (const layer of m.layers || []) { if (layer.type === 'linear') { if (layer.width === oldD) layer.width = game.d; else if (layer.width === oldHidden) layer.width = game.hidden; } }
      for (const m of Object.values(game.modules)) if (m.gated) for (const branch of Object.keys(m.widths)) {
        if (branch === 'downProj' && key === 'd') m.widths[branch] = game.d;
        else if (branch !== 'downProj' && (key === 'd' || key === 'hidden')) m.widths[branch] = game.hidden;
      }
      persist(); render();
    } else if (el.id === 'experimental') { remember(); game.experimental = el.checked; persist(); render(); }
  });
  document.addEventListener('dragstart', event => {
    const block = event.target.closest('[data-block]'), layer = event.target.closest('[data-layer-index]');
    if (block) { event.dataTransfer.setData('text/plain', block.dataset.block); event.dataTransfer.effectAllowed = 'copy'; selected = block.dataset.block; activeTarget = null; updateSelection(); }
    else if (layer && dialog?.type === 'module') { draggingLayer = Number(layer.dataset.layerIndex); event.dataTransfer.setData('application/x-transformer-layer', String(draggingLayer)); event.dataTransfer.effectAllowed = 'move'; }
  });
  document.addEventListener('dragover', event => {
    const slot = event.target.closest('.drop-slot'), layer = event.target.closest('[data-layer-index]');
    if (draggingLayer !== null && layer) { event.preventDefault(); layer.classList.add('drag-over-layer'); }
    else if (slot) { event.preventDefault(); event.dataTransfer.dropEffect = 'copy'; slot.classList.add('drag-over'); }
  });
  document.addEventListener('dragleave', event => { event.target.closest('.drop-slot')?.classList.remove('drag-over'); event.target.closest('[data-layer-index]')?.classList.remove('drag-over-layer'); });
  document.addEventListener('drop', event => {
    const layer = event.target.closest('[data-layer-index]'); if (draggingLayer !== null && layer) { event.preventDefault(); moveLayer(draggingLayer, Number(layer.dataset.layerIndex)); draggingLayer = null; return; }
    const slot = event.target.closest('.drop-slot'); if (slot) { event.preventDefault(); slot.classList.remove('drag-over'); applyBlock(slot.dataset.target, event.dataTransfer.getData('text/plain')); }
  });
  document.addEventListener('dragend', () => { draggingLayer = null; document.querySelectorAll('.drag-over,.drag-over-layer').forEach(el => el.classList.remove('drag-over', 'drag-over-layer')); });
  document.addEventListener('keydown', event => {
    const typing = ['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target.tagName);
    if (event.key === 'Escape') { if (dialog) closeDialog(); else if (traceState) stopTrace(); else { selected = null; activeTarget = null; updateSelection(); } }
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'z' && !typing) { event.preventDefault(); undo(); }
    if (event.key === '/' && !typing && !dialog) { event.preventDefault(); $('#block-search').focus(); }
    if (event.key === 'Tab' && dialog) {
      const items = [...$('.dialog').querySelectorAll('button:not([disabled]), a[href], input, select')].filter(el => el.getClientRects().length), first = items[0], last = items.at(-1);
      if (event.shiftKey && (document.activeElement === first || document.activeElement === $('.dialog'))) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    }
  });
  document.addEventListener('error', event => {
    const img = event.target; if (img.tagName !== 'IMG' || !img.dataset.paperFigure) return;
    if (img.dataset.fallback && !img.dataset.retried) { img.dataset.retried = 'true'; img.src = img.dataset.fallback; }
    else { const message = document.createElement('p'); message.className = 'figure-load-error'; message.textContent = 'Image could not load. Use the PNG download below.'; img.replaceWith(message); }
  }, true);
  render();
})();
