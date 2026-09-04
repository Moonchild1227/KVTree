const COL={GPU:'#73bf69',CPU_PINNED:'#ff9830',EXTERNAL:'#5794f2',
           UNKNOWN:'#8a8f98',MIXED:'#8ab8ff'};
const LBL={GPU:'L1 GPU',CPU_PINNED:'L2 CPU_PINNED',EXTERNAL:'L3 EXTERNAL',
           UNKNOWN:'UNKNOWN'};
const MEDS=['GPU','CPU_PINNED','EXTERNAL','UNKNOWN'];
// One colour per engine instance / attn-dp rank, used by the "by instance"
// grouping. Aggregating everything hides which rank is hot.
const PAL=['#5794f2','#73bf69','#ff9830','#b877d9','#f2cc0c','#e02f44',
           '#8ab8ff','#c0d8a0'];
let groupBy='agg';                  // 'agg' | 'rank'
let logY=false;                     // pool states span 3 orders of magnitude
function streamNames(){
  const n=new Set();
  for(const s of snaps)for(const k in (s.streams||{}))n.add(k);
  return[...n].sort();
}
function rankNames(){
  const n=new Set();
  for(const m of mets)for(const k in (m.ranks||{}))n.add(k);
  return[...n].sort();
}
function shortName(s){const m=String(s).match(/:(\d+)$/);return m?('port '+m[1]):s;}
const PADL=58,PADR=12,PADT=16,PADB=17;   // identical on every time panel
const $=id=>document.getElementById(id);
let snaps=[],trees=[],turns=null,curTree=null,mets=[];
// Grafana semantics: from/to hold EXPRESSIONS, resolved on every draw.
//   null      -> the data boundary (start of everything we have)
//   'now'     -> wall clock at draw time, so the window keeps growing
//   'now-15m' -> relative to wall clock
//   <number>  -> frozen epoch seconds
let range={from:null,to:'now'};
let hidden={'c_pool|free':true};    // "panelId|series" -> true
let hover=null;                     // {ts, src, cx, cy, py}
let sel=null;                       // drag-zoom selection {src,t0,t1}
let follow=true,rfTimer=null,view={x:20,y:14,k:1,touched:false};

const TSP=[{l:'全部 → now (一直累积)',s:-1},
  {l:'Last 5 minutes',s:300},{l:'Last 15 minutes',s:900},
  {l:'Last 30 minutes',s:1800},{l:'Last 1 hour',s:3600},
  {l:'Last 3 hours',s:10800},{l:'Last 6 hours',s:21600},
  {l:'Last 12 hours',s:43200},{l:'Last 24 hours',s:86400},
  {l:'Last 2 days',s:172800},{l:'整个 run (数据范围)',s:0}];

async function j(u){const r=await fetch(u);return r.json();}
function fmtTok(v){
  if(Math.abs(v)>=1048576)return(v/1048576).toFixed(2)+'M';
  if(Math.abs(v)>=1024)return(v/1024).toFixed(1)+'k';
  return''+(Math.round(v*10)/10);
}
function fmtTime(ts,span){
  const d=new Date(ts*1000),p=n=>String(n).padStart(2,'0');
  const hm=p(d.getHours())+':'+p(d.getMinutes());
  if(span===undefined)return hm+':'+p(d.getSeconds());
  if(span>172800)return p(d.getMonth()+1)+'/'+p(d.getDate());
  if(span>43200)return p(d.getMonth()+1)+'/'+p(d.getDate())+' '+hm;
  if(span>600)return hm;
  return hm+':'+p(d.getSeconds());
}
function parseT(s,fb){
  s=(s||'').trim();if(!s)return fb;
  if(/^\d{9,}$/.test(s))return +s;
  const m=s.match(/^(\d{1,2}):(\d{2})(?::(\d{2}))?$/);
  if(m){const d=new Date(dataRange()[0]*1000);
    d.setHours(+m[1],+m[2],+(m[3]||0),0);return d.getTime()/1000;}
  const t=Date.parse(s);return isNaN(t)?fb:t/1000;
}
/* ---- ONE global wall-clock x-axis shared by all time panels ---- */
function dataRange(){
  let lo=Infinity,hi=-Infinity;
  for(const s of snaps){if(s.ts<lo)lo=s.ts;if(s.ts>hi)hi=s.ts;}
  if(turns)for(const t of turns){if(t.sent<lo)lo=t.sent;if(t.recv>hi)hi=t.recv;}
  for(const m of mets){if(m.ts<lo)lo=m.ts;if(m.ts>hi)hi=m.ts;}
  if(!isFinite(lo)){lo=Date.now()/1000-60;hi=lo+60;}
  if(hi-lo<1e-6)hi=lo+1;
  return[lo,hi];
}
/* window that actually has load: turns if we have them, else snapshots.
   The monitor usually keeps sampling long after the replay ends (the engine
   stays up and its cache stays resident), which otherwise squeezes the
   session panel into the left edge while the KV panels look flat-forever. */
function activityRange(){
  if(turns&&turns.length){
    let lo=Infinity,hi=-Infinity;
    for(const t of turns){if(t.sent<lo)lo=t.sent;if(t.recv>hi)hi=t.recv;}
    const pad=Math.max(2,(hi-lo)*0.04);
    return[lo-pad,hi+pad];
  }
  return dataRange();
}
const NOWRE=/^now(?:\s*-\s*(\d+)\s*([smhdw]))?$/i;
const UNIT={s:1,m:60,h:3600,d:86400,w:604800};
function resolveT(v,fallback){
  if(v===null||v===undefined)return fallback;
  if(typeof v==='number')return v;
  const m=String(v).trim().match(NOWRE);
  if(m)return Date.now()/1000-(m[1]?+m[1]*UNIT[m[2].toLowerCase()]:0);
  const p=parseT(String(v),NaN);
  return isNaN(p)?fallback:p;
}
function timeRange(){
  const[a,b]=dataRange();
  let lo=resolveT(range.from,a),hi=resolveT(range.to,b);
  if(hi-lo<1)hi=lo+1;
  return[lo,hi];
}
function xMapper(c){
  const[t0,t1]=timeRange(),W=c.width-PADL-PADR;
  return{X:t=>PADL+(t-t0)/(t1-t0)*W,inv:px=>t0+(px-PADL)/W*(t1-t0),
         t0,t1,W,span:t1-t0};
}
/* ---------------- series builders ---------------- */
function sumStreams(s,f){let v=0;for(const k in (s.streams||{}))v+=f(s.streams[k])||0;
  return v;}
function maxStreams(s,f){let v=0;for(const k in (s.streams||{}))
  v=Math.max(v,f(s.streams[k])||0);return v;}

function perStream(pick){
  return streamNames().map((n,i)=>({name:shortName(n),color:PAL[i%PAL.length],
    vals:snaps.map(s=>[s.ts,pick((s.streams||{})[n])||0])}));
}
// instance is encoded as a shade of the tier colour, so a grouped panel still
// answers "which tier" and "which rank" at the same time
const SHADE=['ff','c0','90','60','40'];
function perStreamMedium(field){
  const names=streamNames(),out=[];
  for(const med of MEDS){
    const any=snaps.some(s=>names.some(
      n=>(((s.streams||{})[n]||{}).mediums||{})[med]));
    if(!any)continue;
    names.forEach((n,i)=>{
      out.push({
        name:LBL[med]+' · '+shortName(n),
        color:COL[med]+SHADE[i%SHADE.length],
        vals:snaps.map(s=>[s.ts,
          ((((s.streams||{})[n]||{}).mediums||{})[med]||{})[field]||0]),
      });
    });
  }
  return out.length?out:perStream(v=>0);
}
function seriesTok(){
  if(groupBy==='rank')return perStreamMedium('tokens');
  return MEDS.map(k=>({name:LBL[k],color:COL[k],
    vals:snaps.map(s=>[s.ts,(s.totals[k]||{}).tokens||0])}));}
function seriesBlk(){
  if(groupBy==='rank')return perStreamMedium('blocks');
  return MEDS.map(k=>({name:LBL[k],color:COL[k],
    vals:snaps.map(s=>[s.ts,(s.totals[k]||{}).blocks||0])}));}

function seriesShape(){
  if(groupBy==='rank')return perStream(v=>v&&v.tree&&v.tree.leaves);
  const g=(f)=>snaps.map(s=>[s.ts,f(s)]);
  return[
   {name:'roots',color:'#5794f2',vals:g(s=>sumStreams(s,x=>x.tree&&x.tree.roots))},
   {name:'leaves',color:'#73bf69',vals:g(s=>sumStreams(s,x=>x.tree&&x.tree.leaves))},
   {name:'branch nodes',color:'#b877d9',
    vals:g(s=>sumStreams(s,x=>x.tree&&x.tree.branch_nodes))},
   {name:'max depth',color:'#ff9830',
    vals:g(s=>maxStreams(s,x=>x.tree&&x.tree.max_depth))},
   {name:'trunk tokens',color:'#f2cc0c',
    vals:g(s=>sumStreams(s,x=>x.tree&&x.tree.trunk_tokens))}];
}

function seriesEvt(){
  if(groupBy==='rank'){
    return streamNames().map((n,i)=>{
      const o=[];
      for(let j=1;j<snaps.length;j++){
        const dt=snaps[j].ts-snaps[j-1].ts;if(dt<=0)continue;
        const a=((snaps[j].streams||{})[n]||{}).stored||0;
        const b=((snaps[j-1].streams||{})[n]||{}).stored||0;
        o.push([snaps[j].ts,Math.max(0,a-b)/dt]);
      }
      return{name:shortName(n)+' stored/s',color:PAL[i%PAL.length],vals:o};
    });
  }
  const rate=key=>{const o=[];
    for(let i=1;i<snaps.length;i++){
      const dt=snaps[i].ts-snaps[i-1].ts;if(dt<=0)continue;
      const d=sumStreams(snaps[i],x=>x[key])-sumStreams(snaps[i-1],x=>x[key]);
      o.push([snaps[i].ts,Math.max(0,d)/dt]);}
    return o;};
  return[{name:'stored/s',color:'#73bf69',vals:rate('stored')},
         {name:'removed/s',color:'#e02f44',vals:rate('removed')},
         {name:'batches/s',color:'#5794f2',vals:rate('batches')},
         {name:'seq gaps',color:'#ff9830',
          vals:snaps.map(s=>[s.ts,sumStreams(s,x=>x.seq_gaps)])}];
}

/* in-flight requests + sessions blocked on a tool call, on a 300-pt grid */
function seriesConc(){
  if(!turns||!turns.length)return[];
  const[t0,t1]=timeRange(),N=300,step=(t1-t0)/N;
  const inf=[],wait=[];
  const bySess=new Map();
  for(const t of turns){
    if(!bySess.has(t.session_id))bySess.set(t.session_id,[]);
    bySess.get(t.session_id).push(t);}
  const gaps=[];
  for(const[,ts]of bySess){ts.sort((a,b)=>a.turn-b.turn);
    for(let i=1;i<ts.length;i++)
      if(ts[i].sent>ts[i-1].recv)gaps.push([ts[i-1].recv,ts[i].sent]);}
  for(let i=0;i<=N;i++){
    const t=t0+i*step;
    inf.push([t,turns.filter(r=>r.sent<=t&&r.recv>=t).length]);
    wait.push([t,gaps.filter(g=>g[0]<=t&&g[1]>=t).length]);}
  return[{name:'in-flight requests',color:'#5794f2',vals:inf},
         {name:'waiting on tool',color:'#ff9830',vals:wait}];
}
function seriesPool(){
  if(!mets.length)return[];
  if(groupBy==='rank'){
    const out=[];
    rankNames().forEach((r,i)=>{
      const c=PAL[i%PAL.length];
      out.push({name:'dp'+r+' used',color:c,
        vals:mets.map(m=>[m.ts,((m.ranks||{})[r]||{}).num_used_tokens||0])});
      out.push({name:'dp'+r+' idle',color:c+'80',
        vals:mets.map(m=>[m.ts,((m.ranks||{})[r]||{}).kv_evictable_tokens||0])});
    });
    return out;
  }
  const g=k=>mets.map(m=>[m.ts,(m.total||{})[k]||0]);
  const out=[
   {name:'used (referenced)',color:'#5794f2',vals:g('num_used_tokens')},
   {name:'evictable (idle L1)',color:'#ff9830',vals:g('kv_evictable_tokens')},
   {name:'free',color:'#3a4350',vals:g('kv_available_tokens')}];
  if(snaps.length)out.push({name:'tree residency (L1)',color:'#73bf69',
    vals:snaps.map(s=>[s.ts,((s.totals.GPU||{}).tokens||0)])});
  return out;
}

/* ---------------- generic time-series panel ---------------- */
const TS=[{c:'c_tok',l:'l_tok',mode:'area',f:seriesTok,unit:' tok'},
          {c:'c_pool',l:'l_pool',mode:'line',f:seriesPool,unit:' tok',log:true},
          {c:'c_blk',l:'l_blk',mode:'area',f:seriesBlk,unit:' blk'},
          {c:'c_shape',l:'l_shape',mode:'line',f:seriesShape,unit:''},
          {c:'c_evt',l:'l_evt',mode:'line',f:seriesEvt,unit:''},
          {c:'c_conc',l:'l_conc',mode:'line',f:seriesConc,unit:''}];

function vis(p,s){return !hidden[p.c+'|'+s.name];}

function axes(x,c,m,max,useLog){
  x.strokeStyle='#23252b';x.lineWidth=1;x.fillStyle='#7b8087';x.font='10px Inter';
  const H=c.height-PADT-PADB;
  for(let i=0;i<=4;i++){
    const y=PADT+H*i/4;
    const v=useLog?Math.pow(10,Math.log10(max+1)*(1-i/4))-1:max*(1-i/4);
    x.beginPath();x.moveTo(PADL,y);x.lineTo(c.width-PADR,y);x.stroke();
    x.textAlign='right';
    x.fillText(fmtTok(v),PADL-6,y+(i===0?8:(i===4?-2:3)));}
  const nt=Math.max(2,Math.floor(m.W/95));
  for(let i=0;i<=nt;i++){
    const t2=m.t0+m.span*i/nt,px=m.X(t2);
    x.beginPath();x.moveTo(px,PADT);x.lineTo(px,c.height-PADB);
    x.strokeStyle='#1b1d22';x.stroke();
    x.textAlign=i===0?'left':(i===nt?'right':'center');
    x.fillText(fmtTime(t2,m.span),px,c.height-4);}
  x.textAlign='center';
}
function drawCursors(x,c,m){
  if(sel&&sel.t1!==sel.t0){
    const a=m.X(Math.min(sel.t0,sel.t1)),b=m.X(Math.max(sel.t0,sel.t1));
    x.fillStyle='#6e9fff22';x.fillRect(a,PADT,b-a,c.height-PADT-PADB);}
  const ts=curTs();
  if(ts&&ts>=m.t0&&ts<=m.t1){
    x.strokeStyle='#ccccdcaa';x.lineWidth=1.5;x.beginPath();
    x.moveTo(m.X(ts),PADT);x.lineTo(m.X(ts),c.height-PADB);x.stroke();}
  if(hover&&hover.ts>=m.t0&&hover.ts<=m.t1){
    const px=m.X(hover.ts);
    x.strokeStyle='#ccccdc';x.lineWidth=1;x.setLineDash([5,4]);x.beginPath();
    x.moveTo(px,PADT);x.lineTo(px,c.height-PADB);x.stroke();x.setLineDash([]);}
}
function legend(p,ss){
  const box=$(p.l);box.innerHTML='';
  for(const s of ss){
    const d=document.createElement('div');
    d.className=vis(p,s)?'':'off';
    d.innerHTML=`<span class="sw" style="background:${s.color}"></span>`+
                `<span>${s.name}</span>`;
    d.onclick=()=>{hidden[p.c+'|'+s.name]=vis(p,s);drawAll();};
    box.appendChild(d);}
}
function nearest(vals,ts){
  let b=null,bd=1e18;
  for(const v of vals){const d=Math.abs(v[0]-ts);if(d<bd){bd=d;b=v;}}
  return b;
}
function noData(x,c,msg){
  x.fillStyle='#5a6068';x.font='12px Inter';x.textAlign='center';
  x.fillText(msg||'No data',(PADL+c.width-PADR)/2,c.height/2);
}
function drawTS(p){
  const c=$(p.c);if(!c)return;
  const x=c.getContext('2d');
  c.width=c.clientWidth;c.height=c.clientHeight;
  x.clearRect(0,0,c.width,c.height);
  // stacking is right for aggregate tiers, wrong once every series is a
  // separate instance -- overlaid lines keep both dimensions readable
  const mode=(groupBy==='rank'&&p.mode==='area')?'line':p.mode;
  const useLog=logY||!!p.log;
  const all=p.f(),ss=all.filter(s=>vis(p,s));
  legend(p,all);
  const m=xMapper(c);
  const pts=all.reduce((a,s)=>a+s.vals.length,0);
  if(!pts){noData(x,c,p.c==='c_conc'?
    'No data — 本次 run 没有 client/turns.jsonl':'No data');return;}
  const inR=v=>v[0]>=m.t0-1e-9&&v[0]<=m.t1+1e-9;
  let max=1;
  if(mode==='area'){
    const n=ss.length?ss[0].vals.length:0;
    for(let i=0;i<n;i++){let sum=0;
      for(const s of ss)if(s.vals[i]&&inR(s.vals[i]))sum+=s.vals[i][1];
      max=Math.max(max,sum);}
  }else for(const s of ss)for(const v of s.vals)if(inR(v))max=Math.max(max,v[1]);
  max*=1.08;
  const H=c.height-PADT-PADB;
  axes(x,c,m,max,useLog);
  x.save();                       // clip: series must never bleed into the axes
  x.beginPath();
  x.rect(PADL,PADT,c.width-PADL-PADR,H);
  x.clip();
  const Y=useLog
    ? v=>PADT+H-Math.log10(Math.max(0,v)+1)/Math.log10(max+1)*H
    : v=>PADT+H-v/max*H;
  if(mode==='area'){
    const n=ss.length?ss[0].vals.length:0,base=new Array(n).fill(0);
    for(const s of ss){
      x.beginPath();let started=false;
      for(let i=0;i<n;i++){const v=s.vals[i];if(!v)continue;
        const px=m.X(v[0]),py=Y(base[i]+v[1]);
        started?x.lineTo(px,py):(x.moveTo(px,py),started=true);}
      for(let i=n-1;i>=0;i--){const v=s.vals[i];if(!v)continue;
        x.lineTo(m.X(v[0]),Y(base[i]));}
      x.closePath();x.fillStyle=s.color+'99';x.fill();
      x.strokeStyle=s.color;x.lineWidth=1;x.stroke();
      for(let i=0;i<n;i++)if(s.vals[i])base[i]+=s.vals[i][1];}
  }else for(const s of ss){
    x.beginPath();let started=false;
    for(const v of s.vals){const px=m.X(v[0]),py=Y(v[1]);
      started?x.lineTo(px,py):(x.moveTo(px,py),started=true);}
    x.strokeStyle=s.color;x.lineWidth=1.4;x.stroke();}
  x.restore();
  drawCursors(x,c,m);
  if(hover&&hover.src===p.c){
    const tip=$('tip');let s='';
    for(const se of ss){const n=nearest(se.vals,hover.ts);
      if(n)s+=`\n${se.name.padEnd(20)} ${fmtTok(n[1])}${p.unit}`;}
    tip.style.display='block';
    tip.textContent=fmtTime(hover.ts)+s;
    tip.style.left=(hover.cx+14)+'px';tip.style.top=(hover.cy+14)+'px';}
}
/* ---------------- session swimlanes ---------------- */
function hitColor(r){
  const st=[[224,47,68],[224,180,0],[115,191,105]];
  const i=r<.5?0:1,f=r<.5?r*2:(r-.5)*2;
  const a=st[i],b=st[i+1];
  return`rgb(${a.map((v,k)=>Math.round(v+(b[k]-v)*f)).join(',')})`;
}
function sessLanes(){
  const l=new Map();
  for(const t of turns){if(!l.has(t.session_id))l.set(t.session_id,[]);
    l.get(t.session_id).push(t);}
  return l;
}
function drawSessions(){
  const c=$('c_sess');if(!c)return;
  const x=c.getContext('2d');
  c.width=c.clientWidth;c.height=c.clientHeight;
  x.clearRect(0,0,c.width,c.height);
  if(!turns||!turns.length){
    noData(x,c,'No data — 本次 run 没有产出 client/turns.jsonl'+
              '(用 --turns 指向同一次 run 的文件)');
    $('sessstats').textContent='';return;}
  const m=xMapper(c),lanes=sessLanes(),ids=[...lanes.keys()];
  x.strokeStyle='#1b1d22';x.fillStyle='#7b8087';x.font='10px Inter';
  const nt=Math.max(2,Math.floor(m.W/95));
  for(let i=0;i<=nt;i++){const t=m.t0+m.span*i/nt,px=m.X(t);
    x.beginPath();x.moveTo(px,PADT);x.lineTo(px,c.height-PADB);x.stroke();
    x.textAlign=i===0?'left':(i===nt?'right':'center');
    x.fillText(fmtTime(t,m.span),px,c.height-4);}
  x.textAlign='center';
  const laneH=Math.min(15,(c.height-PADT-PADB)/Math.max(1,ids.length));
  const barH=Math.max(3,laneH*0.6);
  window._sessLane={laneH,ids,lanes};
  ids.forEach((sid,i)=>{
    const y=PADT+i*laneH+laneH/2,ts=lanes.get(sid).sort((a,b)=>a.turn-b.turn);
    x.strokeStyle='#3a4350';x.lineWidth=1;x.beginPath();
    ts.forEach((t,jj)=>{const cx=m.X((t.sent+t.recv)/2);
      jj?x.lineTo(cx,y):x.moveTo(cx,y);});
    x.stroke();
    for(const t of ts){
      const xa=m.X(t.sent),xb=Math.max(m.X(t.recv),xa+2);
      if(xb<PADL||xa>c.width-PADR)continue;
      const hit=t.prompt_tokens>0?t.cached_tokens/t.prompt_tokens:0;
      x.fillStyle=t.http_status===200?hitColor(hit):'#e02f44';
      x.fillRect(xa,y-barH/2,xb-xa,barH);}
  });
  drawCursors(x,c,m);
  const tool=turns.reduce((a,t)=>a+(t.tool_sleep_ms||0),0)/1000;
  const hits=turns.reduce((a,t)=>a+t.cached_tokens,0)/
             Math.max(1,turns.reduce((a,t)=>a+t.prompt_tokens,0));
  $('sessstats').textContent=`${ids.length} sessions · ${turns.length} turns · `+
    `总等 tool ${tool.toFixed(1)}s · 平均前缀命中 ${(hits*100).toFixed(1)}%`;
  if(hover&&hover.src==='c_sess')sessTip(m);
}
function sessTip(m){
  const s=window._sessLane;if(!s)return;
  const li=Math.floor((hover.py-PADT)/s.laneH),tip=$('tip');
  tip.style.display='block';
  if(li>=0&&li<s.ids.length){
    const sid=s.ids[li];
    const n=s.lanes.get(sid).find(t=>hover.ts>=t.sent-.5&&hover.ts<=t.recv+.5);
    tip.textContent=n?
      `session: ${sid}\nturn: ${n.turn}\n`+
      `prompt: ${n.prompt_tokens} tok (cached ${n.cached_tokens}, `+
      `${(n.prompt_tokens?100*n.cached_tokens/n.prompt_tokens:0).toFixed(1)}%)\n`+
      `completion: ${n.completion_tokens} tok\n`+
      `e2e: ${n.e2e_ms.toFixed(0)} ms   queue: ${n.queue_s.toFixed(2)} s\n`+
      `tool_sleep: ${n.tool_sleep_ms.toFixed(0)} ms   dp_rank: ${n.dp_rank}\n`+
      `finish: ${n.finish_reason}  http: ${n.http_status}`
      :`session: ${sid}\n${fmtTime(hover.ts)}  (间隙: 等 tool / 思考)`;
  }else tip.textContent=fmtTime(hover.ts);
  tip.style.left=(hover.cx+14)+'px';tip.style.top=(hover.cy+14)+'px';
}

/* ---------------- radix tree shape ---------------- */
function curTs(){return trees.length?trees[+$('scrub').value].ts:null;}
function collapseChains(n){
  if(!$('collapse').checked)return n;
  function rec(node){
    let cur=node,count=1,tok=node.tokens,meds=new Set([node.medium]);
    while(cur.children.length===1){
      cur=cur.children[0];count++;tok+=cur.tokens;meds.add(cur.medium);}
    return{hash:node.hash,medium:meds.size===1?node.medium:'MIXED',
           tokens:tok,count,children:cur.children.map(rec)};}
  return rec(n);
}
function drawTree(){
  const g=$('world');g.innerHTML='';
  const st=curTree?curTree.streams[$('stream').value]:null;
  const svg=$('svg');svg.setAttribute('width',svg.parentNode.clientWidth-18);
  if(!st){svg.setAttribute('height',70);
    $('treestats').textContent=trees.length?
      '该时间点没有 block(引擎未接流量)':'本次 run 没有 tree dump';
    return;}
  const roots=st.roots.map(collapseChains);
  const YS=26,XS=110;let leaf=0,maxD=0,nodes=[],links=[];
  function walk(n,d,parent){
    const me={n,d,y:0};
    if(parent)links.push({from:parent,to:me});
    if(!n.children.length)me.y=leaf++;
    else{const cs=n.children.map(c=>walk(c,d+1,me));
      me.y=(cs[0].y+cs[cs.length-1].y)/2;}
    maxD=Math.max(maxD,d);nodes.push(me);return me;}
  roots.forEach(r=>walk(r,0,null));
  const boxH=520,boxW=svg.parentNode.clientWidth-18;
  // Squeeze rows, not the whole drawing: a uniform fit-to-height on 300+ leaves
  // collapses x as well and leaves a thin strip against the left edge. Row
  // pitch shrinks instead, so depth keeps its readable 110px spacing.
  // Node pills are 14px tall, so the row pitch must never go below that or
  // everything overlaps into a green blob. Tall trees scroll inside #treewrap
  // instead of being squeezed; use 适应窗口 for a zoomed-out overview.
  const ysFit=Math.max(18,Math.min(YS,(boxH-40)/Math.max(1,leaf+1)));
  const natW=(maxD+2)*XS,natH=Math.max(140,(leaf+2)*ysFit);
  svg.setAttribute('width',Math.max(boxW,natW));
  svg.setAttribute('height',natH);        // #treewrap scrolls; do not compress
  drawTreeSvg(g,nodes,links,XS,ysFit);
  $('treestats').textContent=
    `${st.blocks} blocks / roots ${st.roots.length} / leaves ${leaf} / `+
    `depth ${maxD+1} / row ${ysFit.toFixed(0)}px`;
  if(!view.touched){
    const k=window._fitAll
      ? Math.min(1,(boxW-30)/Math.max(1,natW),(boxH-30)/natH)
      : Math.min(1,(boxW-30)/Math.max(1,natW));
    view={x:20,y:14,k:Math.max(0.03,k),touched:false};
  }
  applyView();
}
function drawTreeSvg(g,nodes,links,XS,YS){
  const NS='http://www.w3.org/2000/svg';
  for(const l of links){
    const p=document.createElementNS(NS,'path');
    const x1=l.from.d*XS,y1=l.from.y*YS,x2=l.to.d*XS,y2=l.to.y*YS;
    p.setAttribute('class','lnk');
    p.setAttribute('d',`M${x1} ${y1} C ${(x1+x2)/2} ${y1}, `+
                       `${(x1+x2)/2} ${y2}, ${x2} ${y2}`);
    g.appendChild(p);}
  for(const me of nodes){
    const n=me.n,x=me.d*XS,y=me.y*YS,col=COL[n.medium]||COL.UNKNOWN;let el;
    if(n.count>1){
      const w=Math.min(10+n.count*3,90);
      el=document.createElementNS(NS,'rect');
      el.setAttribute('x',x-w/2);el.setAttribute('y',y-7);
      el.setAttribute('width',w);el.setAttribute('height',14);
      el.setAttribute('rx',7);
      const t=document.createElementNS(NS,'text');
      t.setAttribute('x',x);t.setAttribute('y',y+3);
      t.setAttribute('text-anchor','middle');
      t.setAttribute('class','lbl');t.setAttribute('fill','#0b0c0e');
      t.textContent='×'+n.count;g.appendChild(t);
    }else{
      el=document.createElementNS(NS,'circle');
      el.setAttribute('cx',x);el.setAttribute('cy',y);el.setAttribute('r',7);}
    el.setAttribute('class','nd');el.setAttribute('fill',col);
    el.dataset.t=`${n.hash}\nmedium: ${n.medium}\ntokens: ${n.tokens}`+
      (n.count>1?`\nchain: ${n.count} blocks`:'')+
      `\nchildren: ${n.children.length}\ndepth: ${me.d}`;
    g.appendChild(el);}
}

/* ---------------- refresh / draw all ---------------- */
function drawAll(){TS.forEach(drawTS);drawSessions();}
async function refresh(){
  snaps=await j('/api/snapshots');
  trees=await j('/api/trees');
  try{mets=await j('/api/metrics');}catch(e){mets=[];}
  // retry turns while empty: the viewer may have been (re)started with --turns
  // after the page was first opened, and a stale [] would hide the panels
  if(turns===null||!turns.length){
    try{turns=await j('/api/turns');}catch(e){turns=turns||[];}
    if(turns.length){
      $('sesssec').style.display='block';
      if(false){
        const[a,b]=activityRange();range={from:a,to:b};
        $('trlbl').textContent='有负载区间';window._ranged=true;
      }
    }
  }
  const sc=$('scrub');sc.max=Math.max(0,trees.length-1);
  if(follow||+sc.value>trees.length-1)sc.value=trees.length-1;
  if(trees.length){
    $('tslabel').textContent=trees[+sc.value].time;
    curTree=await j('/api/tree?ts='+trees[+sc.value].ts);
    const names=Object.keys(curTree.streams).sort(),sel2=$('stream');
    if(sel2.options.length!==names.length)
      sel2.innerHTML=names.map(n=>`<option>${n}</option>`).join('');
    drawTree();}
  const[a,b]=dataRange();
  $('drange').textContent=fmtTime(a)+' → '+fmtTime(b);
  $('status').textContent=snaps.length?
    `${snaps.length} snapshots · ${trees.length} tree dumps · 末次 `+
    snaps[snaps.length-1].time:'等待数据…';
  drawAll();
}
/* ---------------- interactions ---------------- */
const TSIDS=TS.map(p=>p.c).concat(['c_sess']);
for(const id of TSIDS){
  const c=$(id);if(!c)continue;
  c.addEventListener('mousemove',e=>{
    const r=c.getBoundingClientRect();c.width=c.clientWidth;
    const m=xMapper(c);
    hover={ts:m.inv(e.clientX-r.left),src:id,cx:e.clientX,cy:e.clientY,
           py:e.clientY-r.top};
    if(sel&&sel.src===id)sel.t1=hover.ts;
    drawAll();});
  c.addEventListener('mousedown',e=>{
    const r=c.getBoundingClientRect(),m=xMapper(c);
    const t=m.inv(e.clientX-r.left);sel={src:id,t0:t,t1:t};});
  c.addEventListener('mouseleave',()=>{
    hover=null;$('tip').style.display='none';drawAll();});
}
window.addEventListener('mouseup',()=>{
  if(sel){const a=Math.min(sel.t0,sel.t1),b=Math.max(sel.t0,sel.t1);sel=null;
    if(b-a>0.5){range={from:a,to:b};
      $('trlbl').textContent=fmtTime(a)+' → '+fmtTime(b);}
    drawAll();}
  drag=null;});

$('logy').onchange=e=>{logY=e.target.checked;drawAll();};
$('gb').onchange=e=>{groupBy=e.target.value;
  document.querySelectorAll('[data-agg]').forEach(el=>{
    el.textContent=groupBy==='rank'?el.dataset.rank:el.dataset.agg;});
  drawAll();};
$('tr').onclick=()=>$('pick').classList.toggle('open');
TSP.forEach(q=>{const d=document.createElement('div');d.textContent=q.l;
  d.onclick=()=>{
    if(q.s<0){range={from:null,to:'now'};$('trlbl').textContent='全部 → now';}
    else if(!q.s){range={from:null,to:null};$('trlbl').textContent='整个 run';}
    else{range={from:'now-'+q.s+'s',to:'now'};$('trlbl').textContent=q.l;}
    $('pick').classList.remove('open');drawAll();};
  $('qr').appendChild(d);});
$('papply').onclick=()=>{
  const f=$('pf').value.trim(),g=$('pt').value.trim();
  range={from:f?(NOWRE.test(f)?f:parseT(f,null)):null,
         to:g?(NOWRE.test(g)?g:parseT(g,null)):null};
  $('trlbl').textContent=(f||'数据起点')+' → '+(g||'数据末尾');
  $('pick').classList.remove('open');drawAll();};
$('zout').onclick=()=>{
  const[a,b]=timeRange();
  if(range.to==='now'){   // keep following now, just look further back
    range={from:'now-'+Math.round(2*(b-a))+'s',to:'now'};
    $('trlbl').textContent=`最近 ${Math.round(2*(b-a)/60)} 分钟 → now`;
    drawAll();return;}
  const c=(a+b)/2,h=(b-a);
  range={from:c-h,to:c+h};
  $('trlbl').textContent=fmtTime(range.from)+' → '+fmtTime(range.to);drawAll();};
$('zact').onclick=()=>{const[a,b]=activityRange();range={from:a,to:b};
  $('trlbl').textContent='有负载区间';drawAll();};
$('zrst').onclick=()=>{range={from:null,to:'now'};
  $('trlbl').textContent='全部 → now';drawAll();};
$('now').onclick=refresh;
document.querySelectorAll('.sech').forEach(h=>h.onclick=()=>{
  const g=$(h.dataset.t),open=g.style.display!=='none';
  g.style.display=open?'none':'';
  h.querySelector('.chev').textContent=open?'\u25b6':'\u25bc';
  if(!open)setTimeout(()=>{drawAll();drawTree();},0);});
/* svg zoom / pan */
function applyView(){$('world').setAttribute('transform',
  `translate(${view.x} ${view.y}) scale(${view.k})`);}
let drag=null;
const svgEl=$('svg');
svgEl.addEventListener('wheel',e=>{
  e.preventDefault();
  const r=svgEl.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
  const k2=Math.min(8,Math.max(0.1,view.k*(e.deltaY<0?1.15:1/1.15)));
  view.x=mx-(mx-view.x)*k2/view.k;view.y=my-(my-view.y)*k2/view.k;
  view.k=k2;view.touched=true;applyView();},{passive:false});
svgEl.addEventListener('mousedown',e=>{drag={x:e.clientX,y:e.clientY};
  view.touched=true;});
svgEl.addEventListener('dblclick',()=>{view={x:20,y:14,k:1,touched:true};applyView();});
window.addEventListener('mousemove',e=>{
  if(drag){view.x+=e.clientX-drag.x;view.y+=e.clientY-drag.y;
    drag={x:e.clientX,y:e.clientY};applyView();}
  const t=$('tip');
  if(e.target.dataset&&e.target.dataset.t&&e.target.closest('svg')){
    t.style.display='block';t.textContent=e.target.dataset.t;
    t.style.left=e.clientX+14+'px';t.style.top=e.clientY+14+'px';
  }else if(!hover&&(!e.target.closest||!e.target.closest('canvas')))
    t.style.display='none';});

$('scrub').oninput=async()=>{
  follow=false;$('follow').checked=false;
  $('tslabel').textContent=trees[+$('scrub').value].time;
  curTree=await j('/api/tree?ts='+trees[+$('scrub').value].ts);
  drawTree();drawAll();};
$('follow').onchange=e=>{follow=e.target.checked;if(follow)refresh();};
$('fit').onclick=()=>{window._fitAll=!window._fitAll;
  view.touched=false;drawTree();};
$('collapse').onchange=drawTree;
$('stream').onchange=drawTree;
window.onresize=()=>{drawAll();drawTree();};

function timer(){
  if(rfTimer)clearInterval(rfTimer);
  const s=+$('rf').value;if(!s)return;
  rfTimer=setInterval(()=>{
    if(follow)refresh();
    else j('/api/trees').then(t=>{if(t.length!==trees.length)refresh();});
  },s*1000);}
$('rf').onchange=timer;
refresh();timer();
