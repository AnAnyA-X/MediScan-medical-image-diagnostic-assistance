const fileInput=document.getElementById('fileInput');
const fileName=document.getElementById('fileName');
const analyzeBtn=document.getElementById('analyzeBtn');
const message=document.getElementById('message');
const results=document.getElementById('results');
const apiStatus=document.getElementById('apiStatus');
let selectedFile=null;

fetch('/api/health').then(r=>r.json()).then(data=>{
  apiStatus.textContent=data.model_trained?'AI model ready':'API online · model pending';
}).catch(()=>{apiStatus.textContent='API unavailable';});

fileInput.addEventListener('change',()=>{
  selectedFile=fileInput.files[0]||null;
  fileName.textContent=selectedFile?selectedFile.name:'Choose an X-ray image';
  analyzeBtn.disabled=!selectedFile;
  message.textContent=selectedFile?'Image selected. Ready to analyze.':'Select an image to begin.';
  results.hidden=true;
});
analyzeBtn.addEventListener('click',async()=>{
  if(!selectedFile)return;
  analyzeBtn.disabled=true;analyzeBtn.textContent='Analyzing…';message.textContent='Running model inference…';results.hidden=true;
  const body=new FormData();body.append('file',selectedFile);
  try{
    const response=await fetch('/api/predict',{method:'POST',body});
    const data=await response.json();
    if(!response.ok)throw new Error(data.detail||'Prediction failed.');
    const signalList = Array.isArray(data.results) ? data.results : (data.results && Array.isArray(data.results.results) ? data.results.results : []);
    results.innerHTML='<h3>Model screening signals</h3>'+signalList.slice(0,6).map(x=>
      `<div class="result"><span>${(x.label||'').replaceAll('_',' ')}</span><strong>${((x.probability||0)*100).toFixed(1)}%</strong></div>`
    ).join('')+`<p class="muted">${data.notice||''}</p>`;
    results.hidden=false;message.textContent='Inference complete.';
  }catch(error){message.textContent=error.message;}
  finally{analyzeBtn.disabled=false;analyzeBtn.innerHTML='Analyze image <span>→</span>';}
});
