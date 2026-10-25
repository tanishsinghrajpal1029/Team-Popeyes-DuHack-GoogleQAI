const preview = new URLSearchParams(location.search).has('preview');
const view = {scenario:'crossbar', weak_error_percent:12, block_weak:false};
let topology, current, ui, mode='direct';
let bonusRows=new Map(), currentBonus=null, matrixBonus=false;
const $ = id => document.getElementById(id);
const key = p => `${p[0]},${p[1]}`;
const edgeKey = (a,b) => [a,b].sort().join('|');
const supers='⁰¹²³⁴⁵⁶⁷⁸⁹',subs='₀₁₂₃₄₅₆₇₈₉';
const digits=(value,alphabet)=>String(value).split('').map(d=>alphabet[Number(d)]).join('');

async function requestRoute() {
  $('message').textContent = 'Computing routes…';
  if (preview) {
    try {
      const response = await fetch('/api/route',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(view)});
      const result = await response.json();
      if (!response.ok) throw new Error(result.error);
      show(result);
    } catch (error) { $('message').textContent = `Preview error: ${error.message}`; }
  } else if (ui) {
    ui.send_message('route_request', view);
  }
}

function setView() {
  document.querySelectorAll('[data-scenario]').forEach(button=>button.classList.toggle('selected',button.dataset.scenario===view.scenario));
  $('weak-value').textContent = `${Number(view.weak_error_percent).toFixed(1)}%`;
  requestRoute();
}

function drawMap() {
  if (!topology || !current) return;
  const svg = $('topology');
  const NS='http://www.w3.org/2000/svg';
  svg.replaceChildren();
  const xy = id => {const [r,c]=id.split(',').map(Number);return [37+(c*49),31+(r*48)];};
  const element=(name,attrs,cls)=>{const el=document.createElementNS(NS,name);Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v));if(cls)el.setAttribute('class',cls);svg.append(el);return el;};
  const routes={direct:new Set(),aware:new Set()};
  for (const type of ['direct','aware']) {
    const nodes=current[type].nodes;
    for(let i=0;i<nodes.length-1;i++) routes[type].add(edgeKey(nodes[i],nodes[i+1]));
  }
  for (const [a,b] of topology.couplers) {
    const aa=key(a),bb=key(b),[x1,y1]=xy(aa),[x2,y2]=xy(bb),e=edgeKey(aa,bb);
    let cls='coupler';
    if(routes.aware.has(e))cls+=' aware';
    else if(routes.direct.has(e))cls+=' direct';
    element('line',{x1,y1,x2,y2},cls);
  }
  const directNodes=new Set(current.direct.nodes),awareNodes=new Set(current.aware.nodes);
  for (const point of topology.qubits) {
    const id=key(point),[cx,cy]=xy(id);
    let cls='qubit';
    if(awareNodes.has(id))cls+=' aware';else if(directNodes.has(id))cls+=' direct';
    if(id===current.weak)cls+=current.block_weak?' blocked':' weak';
    if(id===current.start||id===current.end)cls+=' endpoint';
    const circle=element('circle',{cx,cy,r:id===current.weak?8:5},cls);
    const title=document.createElementNS(NS,'title');title.textContent=`Qubit (${id})`;circle.append(title);
  }
  for(const [name,id] of [['START',current.start],['END',current.end],['WEAK',current.weak]]){
    const [x,y]=xy(id);let dx=name==='END'?-43:10,dy=name==='WEAK'?-12:5;
    element('text',{x:x+dx,y:y+dy},'map-label').textContent=name;
  }
}

function drawMatrix() {
  if (!current && !currentBonus) return;
  const box=$('matrix');
  box.replaceChildren();
  if(matrixBonus&&currentBonus){
    const longest=Math.max(currentBonus.candidate_a,currentBonus.candidate_b);
    const widthA=Math.ceil(13*currentBonus.candidate_a/longest);
    const widthB=Math.ceil(13*currentBonus.candidate_b/longest);
    for(let i=0;i<104;i++){
      const row=Math.floor(i/13),col=i%13,pixel=document.createElement('div');
      pixel.className='pixel';
      if((row===1||row===2)&&col<widthA)pixel.classList.add('depth-a');
      if((row===5||row===6)&&col<widthB)pixel.classList.add('depth-b');
      if((row===1||row===2)&&col<widthA&&currentBonus.candidate_a<=currentBonus.candidate_b)pixel.classList.add('depth-winner');
      if((row===5||row===6)&&col<widthB&&currentBonus.candidate_b<=currentBonus.candidate_a)pixel.classList.add('depth-winner');
      box.append(pixel);
    }
    $('matrix-mode').textContent='Upper A · lower B · brighter is shallower';
    $('matrix-mode').style.color='var(--lime)';
    return;
  }
  if(!current)return;
  const m=current.matrix;
  for(let i=0;i<104;i++){
    const row=Math.floor(i/13),col=i%13,bit=1<<col;
    const pixel=document.createElement('div');pixel.className='pixel';
    if(m.nodes[row]&bit)pixel.classList.add('node');
    if(m[mode][row]&bit)pixel.classList.add(mode==='direct'?'path-direct':'path-aware');
    if(i===m.weak_pixel)pixel.classList.add(current.block_weak?'blocked':'weak');
    if(i===m.start_pixel||i===m.end_pixel)pixel.classList.add('endpoint');
    box.append(pixel);
  }
  $('matrix-mode').textContent=mode==='direct'?'Blue · fewest hops':'Green · error-aware';
  $('matrix-mode').style.color=mode==='direct'?'var(--cyan)':'var(--lime)';
}

function show(plan){
  current=plan;
  matrixBonus=false;
  view.scenario=plan.scenario;
  view.weak_error_percent=plan.weak_error_percent;
  view.block_weak=plan.block_weak;
  document.querySelectorAll('[data-scenario]').forEach(button=>button.classList.toggle('selected',button.dataset.scenario===view.scenario));
  $('weak-error').value=String(view.weak_error_percent);
  $('weak-value').textContent=`${Number(view.weak_error_percent).toFixed(1)}%`;
  $('block-weak').checked=view.block_weak;
  drawMap();drawMatrix();
  for(const type of ['direct','aware']){
    $(`${type}-survival`).textContent=`${plan[type].survival_percent.toFixed(1)}%`;
    $(`${type}-hops`).textContent=plan[type].hops;
    $(`${type}-swaps`).textContent=plan[type].swaps;
  }
  const gain=plan.gain_percentage_points;
  $('gain').textContent=`${gain>=0?'+':''}${gain.toFixed(1)} pts`;
  $('insight-copy').textContent=gain>0?`The error-aware route gains ${gain.toFixed(1)} percentage points in this simplified model, even if it needs more SWAPs.`:'Here the shortest route is already competitive under this error setting.';
  $('message').textContent=`Weak qubit (${plan.weak}) · ${plan.board_status||'Ready'}`;
}

function updateBonusView(){
  const N=Number($('bonus-n').value),slider=$('bonus-k');
  slider.max=String(Math.floor(N/2));
  if(Number(slider.value)>Number(slider.max))slider.value=slider.max;
  const n=Number(slider.value),row=bonusRows.get(`${N},${n}`);
  $('bonus-n-value').textContent=N;
  $('bonus-k-value').textContent=n;
  $('circuit-state').textContent=`D${digits(N,supers)}${digits(n,subs)}`;
  $('main-choice').textContent=(N===7&&n===3)||(N===9&&n===4)?'Main task · measured crossover':'Main task · portfolio default';
  if(!row)return;
  const longest=Math.max(row.candidate_a,row.candidate_b);
  $('bonus-a-depth').textContent=row.candidate_a;
  $('bonus-b-depth').textContent=row.candidate_b;
  $('bonus-selected').textContent=row.selected;
  $('bonus-a-bar').style.width=`${100*row.candidate_a/longest}%`;
  $('bonus-b-bar').style.width=`${100*row.candidate_b/longest}%`;
  $('bonus-a-bar').classList.toggle('chosen',row.candidate_a===row.selected);
  $('bonus-b-bar').classList.toggle('chosen',row.candidate_b===row.selected);
}

function showBonus(result){
  currentBonus=result;
  matrixBonus=true;
  drawMatrix();
  $('bonus-message').textContent=`D${digits(result.N,supers)}${digits(result.n,subs)}: ${result.selected} moments selected · ${result.board_status||'Ready'}`;
}

async function requestBonus(){
  const request={N:Number($('bonus-n').value),n:Number($('bonus-k').value)};
  $('bonus-message').textContent='Showing the checked bonus depths…';
  if(preview){
    try{
      const response=await fetch('/api/bonus',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(request)});
      const result=await response.json();
      if(!response.ok)throw new Error(result.error);
      showBonus(result);
    }catch(error){$('bonus-message').textContent=`Preview error: ${error.message}`;}
  }else if(ui){ui.send_message('bonus_request',request);}
}

async function start(){
  for(const button of document.querySelectorAll('[data-scenario]')) button.addEventListener('click',()=>{view.scenario=button.dataset.scenario;setView();});
  $('weak-error').addEventListener('input',event=>{view.weak_error_percent=Number(event.target.value);$('weak-value').textContent=`${view.weak_error_percent.toFixed(1)}%`;});
  $('weak-error').addEventListener('change',setView);
  $('block-weak').addEventListener('change',event=>{view.block_weak=event.target.checked;setView();});
  $('bonus-n').addEventListener('input',updateBonusView);
  $('bonus-k').addEventListener('input',updateBonusView);
  $('show-bonus').addEventListener('click',requestBonus);
  $('show-route').addEventListener('click',requestRoute);
  setInterval(()=>{mode=mode==='direct'?'aware':'direct';drawMatrix();},4200);
  try{
    const response=await fetch('bonus-depths.json');
    if(!response.ok)throw new Error(`HTTP ${response.status}`);
    const data=await response.json();
    bonusRows=new Map(data.cases.map(row=>[`${row.N},${row.n}`,row]));
    updateBonusView();
  }catch(error){$('bonus-message').textContent=`Could not load checked depths: ${error.message}`;}
  try{topology=await (await fetch('topology.json')).json();}catch(error){$('message').textContent=`Could not load topology: ${error.message}`;return;}
  if(preview){$('connection').textContent='Local preview · board disconnected';requestRoute();}
  else{
    ui=new WebUI();
    ui.on_connect(()=>{$('connection').textContent='Connected to UNO Q';requestRoute();});
    ui.on_disconnect(()=>{$('connection').textContent='Board disconnected';});
    ui.on_message('route_result',result=>show(result));
    ui.on_message('route_error',message=>{$('message').textContent=message;});
    ui.on_message('bonus_result',result=>showBonus(result));
    ui.on_message('bonus_error',message=>{$('bonus-message').textContent=message;});
  }
}
start();
