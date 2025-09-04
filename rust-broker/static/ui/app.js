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
  // Get values from the feature grid inputs
  const feats = [
    parseFloat(qs('[name="f0"]').value),
    parseFloat(qs('[name="f1"]').value), 
    parseFloat(qs('[name="f2"]').value),
    parseFloat(qs('[name="f3"]').value)
  ];
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
  // Generate realistic iris values within typical ranges
  const ranges = [
    {min: 4.3, max: 7.9}, // sepal length
    {min: 2.0, max: 4.4}, // sepal width  
    {min: 1.0, max: 6.9}, // petal length
    {min: 0.1, max: 2.5}  // petal width
  ];
  ['f0','f1','f2','f3'].forEach((n, i)=>{ 
    const range = ranges[i];
    const value = (Math.random() * (range.max - range.min) + range.min).toFixed(2);
    qs('[name="' + n + '"]').value = value; 
  });
  console.log('Iris random values set within realistic ranges');
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
  // Generate standardized values (mean=0, std=1) typically in range [-2, +2]
  diabKeys.forEach(k => { 
    const value = (Math.random() * 4 - 2).toFixed(4); // Range: -2.0 to +2.0
    qs('#diab-form [name=' + k + ']').value = value; 
  });
  console.log('Diabetes random standardized values set');
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

// File upload handlers
function setupFileUpload(dropZoneId, fileInputId, formId, endpoint, previewType = 'image') {
  const dropZone = qs('#' + dropZoneId);
  const fileInput = qs('#' + fileInputId);
  const form = qs('#' + formId);
  const submitBtn = form.querySelector('button[type="submit"]');
  const fileInfo = dropZone.querySelector('.file-info');
  const preview = dropZone.querySelector('.' + previewType + '-preview');

  // Click to select file
  dropZone.addEventListener('click', () => fileInput.click());

  // Drag and drop handlers
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelection(files[0]);
    }
  });

  // File input change
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFileSelection(e.target.files[0]);
    }
  });

  function handleFileSelection(file) {
    fileInfo.style.display = 'block';
    fileInfo.textContent = file.name + ' (' + (file.size / 1024).toFixed(1) + ' KB)';
    dropZone.classList.add('has-file');
    submitBtn.disabled = false;

    // Show preview
    if (previewType === 'image' && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        preview.src = e.target.result;
        preview.style.display = 'block';
      };
      reader.readAsDataURL(file);
    } else if (previewType === 'audio' && file.type.startsWith('audio/')) {
      const url = URL.createObjectURL(file);
      preview.src = url;
      preview.style.display = 'block';
    }
  }

  // Form submission
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    if (!fileInput.files[0]) {
      alert('Please select a file first');
      return;
    }

    const formData = new FormData();
    formData.append(previewType, fileInput.files[0]);

    submitBtn.disabled = true;
    submitBtn.textContent = 'Processing...';
    
    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData
      });
      
      const text = await response.text();
      const outputElement = qs('#' + formId.replace('-form', '-out'));
      outputElement.textContent = text;
      console.log(formId + ' submission response:', text);
    } catch (error) {
      const outputElement = qs('#' + formId.replace('-form', '-out'));
      outputElement.textContent = 'Error: ' + error.message;
      console.log(formId + ' submission error:', error);
    } finally {
      submitBtn.disabled = false;
      // Fix button text based on the form type
      if (formId.includes('yolo')) {
        submitBtn.textContent = '🔍 Detect Objects';
      } else if (formId.includes('audio')) {
        submitBtn.textContent = '🎯 Classify Audio';
      } else {
        // Fallback: restore original text
        submitBtn.textContent = submitBtn.textContent.replace('Processing...', 'Submit');
      }
    }
  });
}

// Setup file uploads
setupFileUpload('yolo-drop-zone', 'yolo-file', 'yolo-form', '/predict/yolo', 'image');
setupFileUpload('audio-drop-zone', 'audio-file', 'audio-form', '/predict/audio', 'audio');

// Sample file loading functionality
qs('#yolo-sample').addEventListener('click', async () => {
  const btn = qs('#yolo-sample');
  const originalText = btn.textContent;
  btn.disabled = true;
  btn.textContent = 'Loading...';
  
  try {
    // Try to fetch a small sample image first (dog photo, optimized for object detection)
    let blob;
    try {
      const response = await fetch('https://picsum.photos/400/300.jpg');
      if (response.ok) {
        blob = await response.blob();
      } else {
        throw new Error('Picsum failed');
      }
    } catch {
      // Fallback: Create a simple colored rectangle as test image
      const canvas = document.createElement('canvas');
      canvas.width = 300;
      canvas.height = 200;
      const ctx = canvas.getContext('2d');
      
      // Draw a simple scene with basic shapes (simulating objects for detection)
      ctx.fillStyle = '#87CEEB'; // sky blue
      ctx.fillRect(0, 0, 300, 200);
      
      // Sun (circle)
      ctx.fillStyle = '#FFD700';
      ctx.beginPath();
      ctx.arc(50, 50, 25, 0, 2 * Math.PI);
      ctx.fill();
      
      // Ground
      ctx.fillStyle = '#90EE90';
      ctx.fillRect(0, 150, 300, 50);
      
      // Simple house shape
      ctx.fillStyle = '#8B4513'; // brown
      ctx.fillRect(100, 100, 80, 50);
      ctx.fillStyle = '#FF0000'; // red roof
      ctx.beginPath();
      ctx.moveTo(95, 100);
      ctx.lineTo(140, 70);
      ctx.lineTo(185, 100);
      ctx.closePath();
      ctx.fill();
      
      // Convert canvas to blob
      blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.8));
    }
    
    // Create a File object
    const file = new File([blob], 'sample-image.jpg', { type: 'image/jpeg' });
    
    // Simulate file selection
    const fileInput = qs('#yolo-file');
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    fileInput.files = dataTransfer.files;
    
    // Trigger the change event to update UI
    const changeEvent = new Event('change', { bubbles: true });
    fileInput.dispatchEvent(changeEvent);
    
    console.log('Demo image loaded: ready for object detection');
  } catch (error) {
    console.error('Failed to load sample image:', error);
    alert('Failed to load sample image. Please upload your own image file.');
  } finally {
    btn.disabled = false;
    btn.textContent = originalText;
  }
});

qs('#audio-sample').addEventListener('click', async () => {
  const btn = qs('#audio-sample');
  const originalText = btn.textContent;
  btn.disabled = true;
  btn.textContent = 'Generating...';
  
  try {
    // Generate synthetic audio for classification testing
    
    // Generate a simple synthetic audio sample
    const sampleRate = 16000;
    const duration = 2;
    const numSamples = sampleRate * duration;
    
    // Create a simple beep pattern that Whisper might interpret as speech-like
    const audioBuffer = new ArrayBuffer(44 + numSamples * 2); // WAV header + 16-bit samples
    const view = new DataView(audioBuffer);
    
    // Write WAV header
    const writeString = (offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };
    
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + numSamples * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM format
    view.setUint16(22, 1, true); // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, numSamples * 2, true);
    
    // Generate simple audio pattern (alternating tones to simulate speech)
    let offset = 44;
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      let sample = 0;
      
      // Create a simple pattern that resembles speech frequencies
      if (t < 1.0) {
        sample = 0.3 * Math.sin(2 * Math.PI * 440 * t) * Math.exp(-t * 2);
      } else if (t < 2.0) {
        sample = 0.2 * Math.sin(2 * Math.PI * 330 * t) * Math.exp(-(t-1) * 3);
      }
      
      const intSample = Math.max(-32767, Math.min(32767, Math.floor(sample * 32767)));
      view.setInt16(offset, intSample, true);
      offset += 2;
    }
    
    const blob = new Blob([audioBuffer], { type: 'audio/wav' });
    const file = new File([blob], 'sample-audio.wav', { type: 'audio/wav' });
    
    // Simulate file selection
    const fileInput = qs('#audio-file');
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    fileInput.files = dataTransfer.files;
    
    // Trigger the change event to update UI
    const changeEvent = new Event('change', { bubbles: true });
    fileInput.dispatchEvent(changeEvent);
    
    console.log('Demo audio generated: synthetic audio for classification test');
  } catch (error) {
    console.error('Failed to generate sample audio:', error);
    // Fallback: try to fetch a sample audio file
    try {
      const response = await fetch('https://www2.cs.uic.edu/~i101/SoundFiles/BabyElephantWalk60.wav');
      if (!response.ok) throw new Error('Fallback audio failed');
      
      const blob = await response.blob();
      const file = new File([blob], 'sample-audio.wav', { type: 'audio/wav' });
      
      const fileInput = qs('#audio-file');
      const dataTransfer = new DataTransfer();
      dataTransfer.items.add(file);
      fileInput.files = dataTransfer.files;
      
      const changeEvent = new Event('change', { bubbles: true });
      fileInput.dispatchEvent(changeEvent);
      
      console.log('Sample audio loaded: fallback audio file');
    } catch (fallbackError) {
      console.error('Fallback audio also failed:', fallbackError);
      alert('Failed to load sample audio. Please upload your own audio file.');
    }
  } finally {
    btn.disabled = false;
    btn.textContent = originalText;
  }
});


console.log('UI JavaScript loaded successfully');