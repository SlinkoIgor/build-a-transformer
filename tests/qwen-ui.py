import json, os
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

URL = os.environ.get('GAME_URL', 'http://127.0.0.1:18421/')
CHROME = os.environ.get('CHROME_EXECUTABLE')
ARTIFACTS = Path(os.environ.get('TEST_ARTIFACTS', 'test-results'))
ARTIFACTS.mkdir(parents=True, exist_ok=True)
checks = 0
errors = []
def check(condition, label):
    global checks
    assert condition, label
    checks += 1
    print('PASS:', label, flush=True)

with sync_playwright() as p:
    browser = p.chromium.launch(**({'executable_path': CHROME} if CHROME else {}), headless=True)
    page = browser.new_page(viewport={'width':1280,'height':720}, reduced_motion='reduce', accept_downloads=True)
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.goto(URL)
    # Simulate a completed save from before Qwen was added.
    page.evaluate('''() => {
      const E = TransformerEngine;
      const old = {version:1, active:'bert', games:{}, achievements:{bert:true}};
      for (const key of ['bert','gpt','encdec']) old.games[key] = E.createGame(key);
      E.fillReference(old.games.bert);
      localStorage.setItem('transformer-lab-v1', JSON.stringify(old));
    }''')
    page.reload()
    check(page.locator('.level-tab[data-level="bert"]').get_attribute('aria-current')=='page', 'Old save keeps active model')
    check(page.locator('.level-tab[data-level="bert"] .level-number').count()==1, 'Old achievement preserved')
    page.locator('[data-action="level"][data-level="qwen"]').click()
    def current():
        return page.evaluate('JSON.parse(localStorage.getItem("transformer-lab-v1")).games.qwen')
    def act(name): page.locator(f'[data-action="{name}"]').first.click()
    def place(target, block):
        scope=page.locator('.dialog') if page.locator('.dialog').count() else page.locator('#app')
        button=scope.locator(f'.library-block[data-block="{block}"]').first
        if button.get_attribute('aria-pressed')!='true': button.click()
        scope.locator(f'.drop-slot[data-target="{target}"]').click()
    check(current()['d']==5120 and not current()['structure'], 'Old save acquires an empty Qwen level')
    check(page.locator('.level-tab[data-level="qwen"] .model-year').inner_text()=='2026', 'Qwen creation year')
    check(page.locator('.network-stack').count()==2 and page.locator('.network-board .coarse-slot').count()==14, 'Whole hybrid architecture on one canvas')
    check(page.locator('.hybrid-group>header').inner_text().endswith('× 16 · 64 layers'), 'Hybrid group repeated 16 times')
    check(page.locator('.network-stack>header h2').all_text_contents()==['Layer 4','Layers 1–3'] and page.locator('.network-stack>header .repeat-count').all_text_contents()==['× 1','× 3'], 'Bottom-up three-to-one layout')
    check(page.locator('.network-board .drop-slot').evaluate_all('els=>els.every(e=>e.textContent.includes("Choose a block"))'), 'Empty slots do not disclose the expected module')
    page.screenshot(path=str(ARTIFACTS/'qwen-empty-1280.png'))
    stages=page.evaluate('TransformerEngine.coarseTopology(TransformerEngine.createGame("qwen"))')
    for stage in stages: place('structure::'+stage['id'],stage['expected'])
    act('validate')
    expect(page.locator('.feedback')).to_contain_text('Architecture is correct')
    check(not page.evaluate('TransformerEngine.check(JSON.parse(localStorage.getItem("transformer-lab-v1")).games.qwen).valid'), 'Architecture alone does not complete Qwen')
    page.locator('.assembly-stages [data-stage="inside"]').click()
    slots=page.evaluate('TransformerEngine.topology(TransformerEngine.createGame("qwen"))')
    check(not any(s['id'].endswith('.position') for s in slots), 'No additive position embedding slot')
    for slot in slots: place(slot['id'],slot['expected'])
    metas=page.evaluate('TransformerEngine.modulesFor(TransformerEngine.createGame("qwen"),"decoder")')
    for meta in metas:
        mid=meta['id']
        page.locator(f'[data-action="module"][data-module="{mid}"]').click()
        steps=page.evaluate('(m)=>TransformerEngine.moduleSteps(TransformerEngine.createGame("qwen"),m)',meta)
        if meta['kind']=='delta':
            check(page.locator('.delta-equation').count()==1,'DeltaNet shows causal memory recurrence')
            check(not any(step['expected']=='softmax' for step in steps),'DeltaNet does not use softmax attention')
        if meta['kind']=='ffn': check(page.locator('.gate-branches .branch-card').count()==2,'SwiGLU has independent gate and value branches')
        for step in steps: place(mid+'::'+step['key'],step['expected'])
        act('check-module')
        expect(page.locator('.module-validation')).to_contain_text('Module is correct')
        check(True,mid+': manually assembled internals validate')
        page.screenshot(path=str(ARTIFACTS/(mid.replace('.','-')+'-1280.png')))
        if mid=='decoder.self':
            place(mid+'::gateAct','sigmoid');act('check-module')
            expect(page.locator('.module-validation')).to_contain_text('use SiLU')
            check(True,'Checkpoint swish gate rejects older sigmoid gate')
            place(mid+'::gateAct','silu')
            page.locator('[data-source="k"]').select_option('encoder');act('check-module')
            expect(page.locator('.module-validation')).to_contain_text('current decoder hidden states')
            check(True,'Incorrect K source rejected')
            page.locator('[data-source="k"]').select_option('hidden')
        if mid=='decoder.ffn':
            page.locator('[data-branch-width="upProj"]').fill('5120')
            page.locator('[data-branch-width="upProj"]').press('Tab')
            act('check-module');expect(page.locator('.module-validation')).to_contain_text('17408')
            check(True,'SwiGLU branch-width mismatch detected')
            page.locator('.dialog').focus()
            page.keyboard.press('Meta+z' if CHROME and '/Applications/' in CHROME else 'Control+z')
            check(current()['modules'][mid]['widths']['upProj']==17408,'Undo restores branch width')
        act('close')
    act('validate')
    expect(page.locator('#dialog-title')).to_have_text('Qwen3.8-27B complete!')
    check(page.evaluate('JSON.parse(localStorage.getItem("transformer-lab-v1")).achievements.qwen'),'Complete Qwen earns achievement')
    act('close')
    saved=current()
    page.reload()
    check(current()==saved,'All hybrid operations and widths persist after reload')
    act('settings')
    check(page.locator('[data-config="d"]').input_value()=='5120' and page.locator('[data-config="heads"]').input_value()=='24','Qwen dimensions appear correctly in Settings')
    check('d_head = 256' in page.locator('.head-dimension').inner_text(),'Settings use explicit head width')
    act('close');act('reference')
    expect(page.locator('.paper-overview')).to_contain_text('official model card')
    check(page.locator('.paper-links a').count()==3,'Official card, checkpoint, and source-code links')
    check(page.locator('.paper-links a').evaluate_all('els=>els.every(e=>e.href&&!e.href.includes("undefined"))'),'No missing PDF links')
    check(page.locator('.paper-documents a').count()==2,'Offline architecture source documents')
    with page.expect_download() as download:
        page.locator('.paper-documents a').first.click()
    config=json.loads(Path(download.value.path()).read_text())
    check(config['text_config']['hidden_size']==5120,'Download contains official checkpoint configuration')
    act('close')
    # The main canvas and all workbenches must remain usable at projector and phone widths.
    for width,height in [(1920,1080),(1280,720),(390,844)]:
        page.set_viewport_size({'width':width,'height':height})
        check(page.evaluate('document.documentElement.scrollWidth<=innerWidth'),f'{width}: no page overflow')
        check(page.locator('[data-action="validate"]').is_visible(),f'{width}: validation accessible')
        check(not page.locator('#app').inner_text().count('NaN'),f'{width}: no invalid dimensions')
        page.screenshot(path=str(ARTIFACTS/f'qwen-inside-{width}.png'))
    page.set_viewport_size({'width':1280,'height':720})
    act('settings');act('projector');act('close')
    check(page.evaluate('document.documentElement.scrollWidth<=innerWidth'),'Large text keeps page within viewport')
    check(not errors,'No JavaScript errors: '+str(errors))
    browser.close()
print(f'{checks} Qwen UI checks passed.')
