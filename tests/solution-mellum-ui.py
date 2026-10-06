import json, os
from pathlib import Path
from playwright.sync_api import sync_playwright, expect
URL=os.environ.get('GAME_URL','http://127.0.0.1:18421/')
CHROME=os.environ.get('CHROME_EXECUTABLE')
ARTIFACTS=Path(os.environ.get('TEST_ARTIFACTS','test-results'));ARTIFACTS.mkdir(parents=True,exist_ok=True)
checks=0;errors=[]
def check(condition,label):
 global checks
 assert condition,label
 checks+=1;print('PASS:',label,flush=True)
with sync_playwright() as p:
 browser=p.chromium.launch(**({'executable_path':CHROME} if CHROME else {}),headless=True)
 page=browser.new_page(viewport={'width':1280,'height':720},reduced_motion='reduce',accept_downloads=True)
 page.on('pageerror',lambda e:errors.append(str(e)));page.goto(URL)
 def act(name):page.locator(f'[data-action="{name}"]').first.click()
 def state():return page.evaluate('JSON.parse(localStorage.getItem("transformer-lab-v1"))')
 def current():s=state();return s['games'][s['active']]
 def normalized(g):return page.evaluate('(g)=>{TransformerEngine.progress(g);return g}',g)
 def level(name):page.locator(f'.level-tab[data-level="{name}"]').click()
 def solution():page.locator('.workspace-footer [data-action="show-solution"]').click()
 def place(target,block):
  scope=page.locator('.dialog') if page.locator('.dialog').count() else page.locator('#app')
  button=scope.locator(f'.library-block[data-block="{block}"]').first
  if button.get_attribute('aria-pressed')!='true':button.click()
  scope.locator(f'.drop-slot[data-target="{target}"]').click()
 for name in ['bert','gpt','encdec','qwen','mellum']:
  level(name);before=current();others={k:v for k,v in state()['games'].items() if k!=name}
  solution()
  check(not page.locator('.dialog').count(),name+': solution opens immediately without confirmation')
  check(page.evaluate('(g)=>TransformerEngine.check(g).valid',current()),name+': every module and internal operation is correct')
  check(page.evaluate('(g)=>TransformerEngine.progress(g).percent',current())==100,name+': solution fills 100 percent')
  check(not state()['achievements'].get(name),name+': revealing solution does not award completion')
  check({k:v for k,v in state()['games'].items() if k!=name}==others,name+': other models preserved')
  act('undo');check(normalized(current())==normalized(before),name+': one undo restores the exact previous build')
 # A correct answer restores original settings, even after invalid experiments.
 level('qwen');page.evaluate('''()=>{const s=JSON.parse(localStorage.getItem('transformer-lab-v1'));Object.assign(s.games.qwen,{norm:'post',experimental:true,d:128,heads:1,hidden:256,depth:2});s.games.qwen.structure={'decoder.self':'recurrent'};s.games.qwen.modules={};localStorage.setItem('transformer-lab-v1',JSON.stringify(s));}''');page.reload()
 before=current();solution();g=current()
 check(g['norm']=='pre' and not g['experimental'] and (g['d'],g['heads'],g['hidden'],g['depth'])==(5120,24,17408,64),'Solution restores original normalization and all dimensions')
 check(page.evaluate('(g)=>TransformerEngine.check(g).valid',g),'Invalid experiment becomes the actual correct model')
 act('undo');check(normalized(current())==normalized(before),'Undo also restores experiment settings and wrong placements')
 # Assemble both Mellum layer types and their experts manually.
 level('mellum');check(page.locator('.model-year').last.inner_text()=='2026','Mellum year')
 check(page.locator('.hybrid-group>header .repeat-count').inner_text()=='× 7 · 28 layers','Seven groups of four layers')
 check(page.locator('.network-board .coarse-slot').count()==14,'Both layer types shown together')
 stages=page.evaluate('TransformerEngine.coarseTopology(TransformerEngine.createGame("mellum"))')
 for st in stages:place('structure::'+st['id'],st['expected'])
 act('validate');expect(page.locator('.feedback')).to_contain_text('Architecture is correct')
 page.locator('.assembly-stages [data-stage="inside"]').click()
 for slot in page.evaluate('TransformerEngine.topology(TransformerEngine.createGame("mellum"))'):place(slot['id'],slot['expected'])
 metas=page.evaluate('TransformerEngine.modulesFor(TransformerEngine.createGame("mellum"),"decoder")')
 for meta in metas:
  mid=meta['id'];page.locator(f'[data-action="module"][data-module="{mid}"]').click()
  steps=page.evaluate('(m)=>TransformerEngine.moduleSteps(TransformerEngine.createGame("mellum"),m)',meta)
  for step in steps:place(mid+'::'+step['key'],step['expected'])
  act('check-module');expect(page.locator('.module-validation')).to_contain_text('Module is correct')
  check(True,mid+': manual internals validate')
  if mid=='decoder.local':
   check('Toy window: 3 tokens; actual model window: 1024' in page.locator('.mask-demo').inner_text(),'Sliding preview labels its miniature window accurately')
   page.locator('[data-action="mask-query"][data-query="5"]').click()
   check(page.locator('.matrix-cell.query-row.allowed').count()==3,'Sliding mask excludes old tokens and future tokens')
   place(mid+'::mask','causal');act('check-module');expect(page.locator('.module-validation')).to_contain_text('sliding causal mask')
   check(True,'Full causal mask rejected for local layer');place(mid+'::mask','sliding-causal')
  if mid=='decoder.self':
   check(page.locator('[data-target="decoder.self::gateAct"]').count()==0,'Mellum attention has no output gate')
   place(mid+'::rope','rope');act('check-module');expect(page.locator('.module-validation')).to_contain_text('RoPE + YaRN')
   check(True,'Full attention requires YaRN-scaled positions');place(mid+'::rope','rope-yarn')
  if meta['kind']=='moe':
   check(page.locator('.gate-branches .branch-card').count()==2,'Expert SwiGLU has independent parallel branches')
   page.locator('[data-routing="topK"]').select_option('4');act('check-module');expect(page.locator('.module-validation')).to_contain_text('selects 8')
   check(True,'Incorrect active expert count rejected');page.locator('[data-routing="topK"]').select_option('8')
   page.locator('[data-branch-width="upProj"]').fill('7168');page.locator('[data-branch-width="upProj"]').press('Tab');act('check-module');expect(page.locator('.module-validation')).to_contain_text('896')
   check(True,'Dense FFN width rejected as expert width')
   page.locator('[data-branch-width="upProj"]').fill('896');page.locator('[data-branch-width="upProj"]').press('Tab')
  page.locator('.workbench-main').evaluate('el=>el.scrollTop=0');page.screenshot(path=str(ARTIFACTS/(mid.replace('.','-')+'.png')))
  act('close')
 check(page.evaluate('(g)=>TransformerEngine.check(g).valid',current()),'Manually assembled Mellum is complete')
 # Keep inside view, fill a missing operation, and allow undo.
 page.locator('[data-action="remove"][data-target="decoder.finalNorm"]').click();partial=current();solution()
 check(page.locator('.assembly-stages [data-stage="inside"]').get_attribute('aria-pressed')=='true','Solution keeps the chosen inside view')
 check(page.evaluate('(g)=>TransformerEngine.check(g).valid',current()),'Solution fills a removed internal operation')
 act('undo');check(current()==partial,'Undo restores the partial internal build');solution();saved=current();page.reload();check(current()==saved,'Revealed solution persists on reload')
 act('reference');check(page.locator('.paper-links a').count()==3,'Mellum links to official model card, pinned config, and implementation')
 check(page.locator('.paper-documents a').count()==2,'Mellum source documents work offline')
 with page.expect_download() as dl:page.locator('.paper-documents a').first.click()
 config=json.loads(Path(dl.value.path()).read_text());check(config['num_experts']==64 and config['num_experts_per_tok']==8,'Download is the original Mellum config');act('close')
 for width,height in [(1920,1080),(1280,720),(1024,768),(390,844)]:
  page.set_viewport_size({'width':width,'height':height})
  check(page.evaluate('document.documentElement.scrollWidth<=innerWidth'),f'{width}: no horizontal page overflow')
  check(page.locator('.workspace-footer [data-action="show-solution"]').is_visible(),f'{width}: correct-answer button accessible')
  check(page.locator('[data-action="validate"]').is_visible(),f'{width}: check button accessible')
  page.screenshot(path=str(ARTIFACTS/f'mellum-inside-{width}.png'))
 page.set_viewport_size({'width':1280,'height':720});act('settings');act('projector');act('close')
 check(page.evaluate('document.documentElement.scrollWidth<=innerWidth'),'Large text keeps all five model tabs within page')
 check(not errors,'No JavaScript errors: '+str(errors));browser.close()
print(f'{checks} correct-solution and Mellum UI checks passed.')
