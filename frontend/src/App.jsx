import React, { useEffect, useRef, useState } from 'react';
import { Activity, ArrowRight, Check, ChevronRight, FileImage, Image as ImageIcon, Menu, Moon, RefreshCw, ScanLine, Server, Sparkles, Sun, Trash2, Upload, X } from 'lucide-react';

const API_BASE = (import.meta.env.VITE_API_URL || 'http://localhost:8001').replace(/\/$/, '');
const REQUEST_TIMEOUT = 30000;
async function apiFetch(path, options = {}, timeout = REQUEST_TIMEOUT) {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(`${API_BASE}${path}`, { ...options, signal: controller.signal });
    const contentType = response.headers.get('content-type') || '';
    const payload = contentType.includes('application/json') ? await response.json() : await response.text();
    if (!response.ok) {
      const message = typeof payload === 'object' ? payload.detail : payload;
      throw new Error(message || `API request failed (${response.status}).`);
    }
    if (typeof payload !== 'object') throw new Error('The AI server returned an invalid response.');
    return payload;
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('The AI server took too long to respond. Please try again.');
    if (error instanceof TypeError) throw new Error('AI Engine Offline. The cloud API could not be reached.');
    throw error;
  } finally { window.clearTimeout(timer); }
}
const CLASSES = ['Healthy', 'Powdery', 'Rust'];
const formatBytes = bytes => bytes < 1048576 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / 1048576).toFixed(1)} MB`;

function Navbar({ theme, setTheme, online }) {
  const [open, setOpen] = useState(false);
  return <nav className="nav shell">
    <a className="logo" href="#top"><span className="logo-mark"><ScanLine size={17}/></span>AI<span>VISION</span></a>
    <div className={`nav-links ${open ? 'open' : ''}`}><a href="#overview">Overview</a><a href="#vision">Vision</a><a href="#models">Models</a><a href="#about">About</a></div>
    <div className="nav-actions"><div className={`model-pill ${online ? 'online' : ''}`}><i/>Model {online ? 'Ready' : 'Offline'}</div>
      <button className="icon-button" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} aria-label="Toggle theme">{theme === 'dark' ? <Sun size={17}/> : <Moon size={17}/>}</button>
      <button className="icon-button menu-button" onClick={() => setOpen(!open)} aria-label="Open menu"><Menu size={18}/></button></div>
  </nav>;
}

function Hero() { return <section className="hero shell" id="overview"><div className="eyebrow"><Sparkles size={14}/> AI_VISION <b>•</b> TensorFlow Lite <b>•</b> 3 Classes</div><h1>Vision Intelligence,<br/><span>Simplified.</span></h1><p>Run image inference using a lightweight TensorFlow Lite computer vision model.</p><a className="hero-link" href="#vision">Open inference workspace <ArrowRight size={16}/></a></section>; }

function ImageUploader({ file, preview, loading, onFile, onRemove, onAnalyze, error }) {
  const input = useRef(null), [dragging, setDragging] = useState(false);
  return <div className="workspace-card"><div className="card-heading"><div><span className="kicker">01 / Input</span><h2>Image Input</h2></div><FileImage size={20}/></div>
    {!file ? <div className={`dropzone ${dragging ? 'dragging' : ''}`} onClick={() => input.current?.click()} onDragOver={e => {e.preventDefault();setDragging(true)}} onDragLeave={() => setDragging(false)} onDrop={e => {e.preventDefault();setDragging(false);onFile(e.dataTransfer.files?.[0])}}><div className="upload-icon"><Upload size={22}/></div><h3>Drop an image here</h3><p>or <span>browse from your device</span></p><small>JPG&nbsp; • &nbsp;JPEG&nbsp; • &nbsp;PNG</small><input ref={input} hidden type="file" accept="image/jpeg,image/png" onChange={e => onFile(e.target.files?.[0])}/></div>
    : <div className={`image-preview ${loading ? 'scanning' : ''}`}><img src={preview} alt="Selected upload preview"/>{loading && <div className="scan-line"/>}<div className="image-meta"><div><ImageIcon size={17}/><span><strong>{file.name}</strong><small>{formatBytes(file.size)}</small></span></div><div className="preview-actions"><button onClick={() => input.current?.click()}><RefreshCw size={14}/> Replace</button><button onClick={onRemove}><Trash2 size={14}/> Remove</button></div></div><input ref={input} hidden type="file" accept="image/jpeg,image/png" onChange={e => onFile(e.target.files?.[0])}/></div>}
    {error && <div className="error-message"><X size={16}/><span>{error}</span></div>}<button className="primary-button" disabled={!file || loading} onClick={onAnalyze}>{loading ? <><span className="pulse"/>Analyzing...</> : <><Sparkles size={17}/>Analyze Image<ChevronRight size={16}/></>}</button>
  </div>;
}

function PredictionResult({ result, loading, online }) { return <div className="workspace-card"><div className="card-heading"><div><span className="kicker">02 / Output</span><h2>AI Analysis</h2></div><Activity size={20}/></div>
  {loading ? <div className="result-skeleton"><div/><div/><div/><div/><div/></div> : result ? <div className="result-content"><div className="prediction-summary"><div><span>Predicted Class</span><strong className={`class-${result.prediction.toLowerCase()}`}>{result.prediction}</strong></div><div><span>Confidence</span><strong>{Number(result.confidence).toFixed(2)}%</strong></div></div><div className="probabilities"><h3>Class Probabilities</h3>{CLASSES.map(name => { const value = Number(result.probabilities?.[name] || 0); return <div className="probability" key={name}><div><span><i className={`dot-${name.toLowerCase()}`}/>{name}</span><b>{value.toFixed(2)}%</b></div><div className="bar"><span className={`fill-${name.toLowerCase()}`} style={{'--width':`${value}%`}}/></div></div>})}</div><div className="verified"><Check size={14}/> Inference completed by AI_VISION.tflite</div></div>
  : <div className="empty-state"><div className="empty-visual"><ScanLine size={28}/><i/><i/><i/></div><h3>{online ? 'No analysis yet' : 'AI Engine Offline'}</h3><p>{online ? 'Upload an image and run AI Vision to see prediction results.' : 'Start the inference backend to enable image analysis.'}</p></div>}</div>; }

function ModelSections() { const specs=[['Architecture','CNN'],['Input','224 × 224 RGB'],['Classes','3'],['Runtime','TensorFlow Lite']]; return <><section className="section shell" id="models"><div className="section-title"><span>Model specification</span><h2>Model Information</h2><p>A compact vision model optimized for lightweight image inference.</p></div><div className="metric-grid">{specs.map(([a,b],i)=><div className="metric-card" key={a}><span>0{i+1}</span><p>{a}</p><strong>{b}</strong></div>)}</div><div className="class-row"><span>Classes</span><div><i className="dot-healthy"/>Healthy <i className="dot-powdery"/>Powdery <i className="dot-rust"/>Rust</div></div></section>
  <section className="section shell"><div className="section-title compact"><span>Evaluation</span><h2>Model Performance</h2></div><div className="performance-grid">{[['Test Accuracy','82.80%'],['Input Resolution','224 × 224'],['Output Classes','3'],['Model Format','TFLite']].map(([a,b])=><div key={a}><span>{a}</span><strong>{b}</strong></div>)}</div><p className="note">Accuracy is measured on the test dataset. Performance on external images may vary depending on image quality and distribution.</p></section></>; }

function InferenceFlow() { const steps=[['Upload Image','Image'],['Preprocessing','224×224 RGB'],['AI_VISION','AI Model'],['Inference','TensorFlow Lite'],['Prediction','3-class output']]; return <section className="section shell"><div className="section-title"><span>Pipeline</span><h2>How it works</h2><p>From source image to model output in five focused steps.</p></div><div className="flow">{steps.map(([a,b],i)=><React.Fragment key={a}><div className="flow-step"><b>{String(i+1).padStart(2,'0')}</b><strong>{a}</strong><span>{b}</span></div>{i<4&&<ArrowRight className="flow-arrow" size={18}/>}</React.Fragment>)}</div></section>; }

function StatusAndHistory({ online, history, clearHistory }) { return <section className="section shell split-section"><div><div className="section-title compact"><span>Runtime</span><h2>System Status</h2></div><div className="status-panel"><div><span>AI Engine</span><b className={online?'online-text':'offline-text'}><i/>{online?'Online':'Offline'}</b></div><div><span>Model</span><b>AI_VISION.tflite</b></div><div><span>Input Shape</span><b>1 × 224 × 224 × 3</b></div><div><span>Output Classes</span><b>3</b></div><div><span>Runtime</span><b>TensorFlow Lite</b></div></div></div>
  <div><div className="section-title compact history-heading"><div><span>Session</span><h2>Analysis History</h2></div>{history.length>0&&<button onClick={clearHistory}>Clear</button>}</div><div className="history-panel">{history.length?<div className="history-table"><div className="history-row head"><span>Image</span><span>Prediction</span><span>Confidence</span><span>Time</span></div>{history.map(x=><div className="history-row" key={x.id}><span title={x.name}>{x.name}</span><span className={`class-${x.prediction.toLowerCase()}`}>{x.prediction}</span><span>{Number(x.confidence).toFixed(2)}%</span><span>{x.time}</span></div>)}</div>:<div className="history-empty"><Server size={24}/><h3>No predictions yet.</h3><p>Your recent AI analyses will appear here.</p></div>}</div></div></section>; }

export default function App() {
  const [theme,setTheme]=useState(()=>localStorage.getItem('ai-theme')||'dark'),[online,setOnline]=useState(false),[file,setFile]=useState(null),[preview,setPreview]=useState(null),[loading,setLoading]=useState(false),[result,setResult]=useState(null),[error,setError]=useState('');
  const [history,setHistory]=useState(()=>{try{return JSON.parse(sessionStorage.getItem('ai-history'))||[]}catch{return[]}});
  useEffect(()=>{document.documentElement.dataset.theme=theme;localStorage.setItem('ai-theme',theme)},[theme]);
  useEffect(()=>{apiFetch('/health',{},10000).then(data=>setOnline(data.status==='online')).catch(()=>setOnline(false))},[]);
  const chooseFile=selected=>{if(!selected)return;if(!['image/jpeg','image/png'].includes(selected.type)){setError('Please select a valid JPG, JPEG, or PNG image.');return}if(preview)URL.revokeObjectURL(preview);setFile(selected);setPreview(URL.createObjectURL(selected));setResult(null);setError('')};
  const remove=()=>{if(preview)URL.revokeObjectURL(preview);setFile(null);setPreview(null);setResult(null);setError('')};
  const analyze=async()=>{if(!file)return;setLoading(true);setError('');setResult(null);const body=new FormData();body.append('file',file);try{const data=await apiFetch('/predict',{method:'POST',body});setResult(data);setOnline(true);const item={id:Date.now(),name:file.name,prediction:data.prediction,confidence:data.confidence,time:new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'})};setHistory(prev=>{const next=[item,...prev].slice(0,6);sessionStorage.setItem('ai-history',JSON.stringify(next));return next})}catch(e){setOnline(false);setError(e.message||'Image analysis failed. Please try again.')}finally{setLoading(false)}};
  return <div id="top"><Navbar theme={theme} setTheme={setTheme} online={online}/><main><Hero/><section className="workspace-section shell" id="vision"><div className="workspace-intro"><div><span>Inference workspace</span><h2>Test the model</h2></div><p>Upload a supported image and inspect the complete probability distribution from the live model.</p></div><div className="workspace-grid"><ImageUploader file={file} preview={preview} loading={loading} onFile={chooseFile} onRemove={remove} onAnalyze={analyze} error={error}/><PredictionResult result={result} loading={loading} online={online}/></div></section><ModelSections/><InferenceFlow/><StatusAndHistory online={online} history={history} clearHistory={()=>{sessionStorage.removeItem('ai-history');setHistory([])}}/></main><footer id="about"><div className="shell footer-inner"><div><a className="logo" href="#top"><span className="logo-mark"><ScanLine size={17}/></span>AI<span>VISION</span></a><p>Experimental Computer Vision Model</p></div><div className="footer-copy"><p>Built for AI experimentation and learning.</p><small>AI predictions are experimental and should not be considered a definitive diagnosis.</small></div></div></footer></div>;
}
