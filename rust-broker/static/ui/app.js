const qs = (s)=>document.querySelector(s);
const jobsTbody = qs('#jobs');
const selectedDiv = qs('#selected');
const jobJson = qs('#job-json');
const purgeBtn = qs('#purge');
const purgeStatus = qs('#purge-status');
let selectedId = null;

function ell(s){ return s.length>12 ? s.slice(0,8)+'…'+s.slice(-4) : s }
function badge(status){
  if(status==='completed') return '<span class="ok">completed</span>';
  if(status==='failed') return '<span class="fail">failed</span>';
  return '<span class="queued">'+status+'</span>';
}
function summarizeResult(r, err){
  if(err) return '<span class="fail">'+err+'</span>';
  if(!r) return '<span class="muted">—</span>';
  try{
    if(r.type==='classification' && r.label_name!==undefined){
      return 'class: <strong>' + r.label_name + '</strong> (id=' + (r.label_id ?? '?') + ')';
    }
    if(r.type==='regression' && r.prediction!==undefined){
      const p = Number(r.prediction);
      return 'ŷ = <strong>' + (isFinite(p)? p.toFixed(3): p) + '</strong>';
    }
  }catch(e){}
  let s = JSON.stringify(r);
  if(s.length>96) s = s.slice(0,93)+'…';
  return s;
}

async function listJobs(){
  try{
    const r = await fetch('/jobs');
    const data = await r.json();
    jobsTbody.innerHTML = data.map(j =>
      '<tr>' +
        '<td><a href="#" data-job-id="' + j.id + '"><code title="' + j.id + '">' + ell(j.id) + '</code></a></td>' +
        '<td>' + j.model + '</td>' +
        '<td>' + badge(j.status) + '</td>' +
        '<td class="result">' + summarizeResult(j.result, j.error) + '</td>' +
        '<td class="nowrap"><small>' + j.updated_at + '</small></td>' +
      '</tr>').join('');
    // bind click handlers
    document.querySelectorAll('[data-job-id]').forEach(a=>{
      a.addEventListener('click', ev=>{
        ev.preventDefault();
        selectedId = a.getAttribute('data-job-id');
        loadJob(selectedId);
      });
    });
  }catch(e){ console.log('listJobs error:', e); }
}
async function loadJob(id){
  try{
    const r = await fetch('/jobs/'+id);
    const data = await r.json();
    selectedDiv.innerHTML = '<div><strong>Job:</strong> <code>' + data.id + '</code> • <strong>Model:</strong> ' + data.model + ' • <strong>Status:</strong> ' + data.status + '</div>';
    jobJson.style.display = 'block';
    jobJson.textContent = JSON.stringify(data, null, 2);
  }catch(e){
    selectedDiv.innerHTML = '<span class="fail">Failed to load job ' + id + '</span>';
    jobJson.style.display = 'none';
    console.log('loadJob error:', e);
  }
}
setInterval(()=>{
  listJobs();
  if(selectedId) loadJob(selectedId);
}, 2000);
listJobs();

// Iris form
qs('#iris-form').addEventListener('submit', async (ev)=>{
  ev.preventDefault();
  const f = ev.target;
  const feats = [f.f0.value, f.f1.value, f.f2.value, f.f3.value].map(parseFloat);
  try{
    const r = await fetch('/predict/iris', {method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify({features:feats})});
    const text = await r.text();
    qs('#iris-out').textContent = text;
    console.log('Iris submission response:', text);
  }catch(e){
    qs('#iris-out').textContent = 'Error: ' + e.message;
    console.log('Iris submission error:', e);
  }
});
qs('#iris-rand').addEventListener('click', ()=>{
  const rand = ()=> (Math.random()*6).toFixed(2);
  ['f0','f1','f2','f3'].forEach((n)=>{ qs('#iris-form [name=' + n + ']').value = rand(); });
  console.log('Iris random values set');
});

// Diabetes form (build inputs)
const diabKeys = ["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"];
qs('#diab-fields').innerHTML = diabKeys.map(k=>'<label>' + k + ': <input type="number" step="0.0001" name="' + k + '" value="0.0"/></label>').join('<br/>');
qs('#diab-form').addEventListener('submit', async (ev)=>{
  ev.preventDefault();
  const f = ev.target;
  const features = Object.fromEntries(diabKeys.map(k=>[k, parseFloat(f[k].value)]));
  try{
    const r = await fetch('/predict/diabetes', {method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify({features})});
    const text = await r.text();
    qs('#diab-out').textContent = text;
    console.log('Diabetes submission response:', text);
  }catch(e){
    qs('#diab-out').textContent = 'Error: ' + e.message;
    console.log('Diabetes submission error:', e);
  }
});
qs('#diab-rand').addEventListener('click', ()=>{
  diabKeys.forEach(k => { qs('#diab-form [name=' + k + ']').value = (Math.random()*0.2 - 0.1).toFixed(4); });
  console.log('Diabetes random values set');
});

// Purge DB & Queue
purgeBtn.addEventListener('click', async ()=>{
  if(!confirm('Purge ALL jobs and clear the queue? This cannot be undone.')) return;
  purgeBtn.disabled = true;
  purgeStatus.textContent = 'Purging…';
  try{
    const r = await fetch('/admin/purge', { method: 'POST' });
    const data = await r.json();
    purgeStatus.textContent = 'Purged: jobs=' + data.deleted_jobs + ', queue_msgs=' + data.purged_messages;
    // refresh lists
    selectedId = null;
    jobJson.style.display = 'none';
    jobJson.textContent = '';
    await listJobs();
    console.log('Purge completed:', data);
  }catch(e){
    purgeStatus.textContent = 'Purge failed';
    console.log('Purge error:', e);
  }finally{
    purgeBtn.disabled = false;
    setTimeout(()=> purgeStatus.textContent = '', 5000);
  }
});

console.log('UI JavaScript loaded successfully');