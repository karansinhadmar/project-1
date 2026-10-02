import { useEffect, useRef, useState } from "react";
import sampleMriImage from "./sample.jpeg";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";
const ALLOWED = ["image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"];
const MAX_MB = 20;

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [online, setOnline] = useState(null);
  const [showInfo, setShowInfo] = useState(true);
  const [showSample, setShowSample] = useState(false);
  
  const inputRef = useRef(null);

  useEffect(() => {
    fetch(`${API}/health`).then((r) => setOnline(r.ok)).catch(() => setOnline(false));
  }, []);

  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview]);

  function choose(f) {
    setResult(null);
    setError("");
    if (!f) return;
    if (!ALLOWED.includes(f.type)) return setError("Please upload a JPG, PNG, BMP, TIFF or WEBP image.");
    if (f.size > MAX_MB * 1024 * 1024) return setError(`File is larger than ${MAX_MB} MB.`);
    setFile(f);
    setPreview(URL.createObjectURL(f));
  }

  async function analyze() {
    if (!file) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const body = new FormData();
      body.append("file", file);
      const res = await fetch(`${API}/predict`, { method: "POST", body });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || `Server error (${res.status})`);
      setResult(data);
    } catch (e) {
      setError(e.message === "Failed to fetch" ? "Cannot reach the backend. Is it running on port 8000?" : e.message);
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError("");
    if (inputRef.current) inputRef.current.value = "";
  }

  const sorted = result ? Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]) : [];

  return (
    <main className={`container ${result ? "has-results" : ""}`}>
      <header>
        <h1>🧠 Alzheimer's Disease Detection</h1>
        <p className="sub">DenseNet-121 · Brain MRI · 4-class classification</p>
        <span className={`status ${online === null ? "" : online ? "ok" : "bad"}`}>
          {online === null ? "Checking server…" : online ? "Server online" : "Server offline"}
        </span>
      </header>

      {/* Hide info & sample boxes only when results are active */}
      {!result && (
        <>
          {/* Info Box: About Alzheimer's & MRI Stages */}
          <section className="card info-box">
            <div className="info-header" onClick={() => setShowInfo((prev) => !prev)}>
              <h2>📖 About Alzheimer's & Classification Stages</h2>
              <button type="button" className="toggle-btn" aria-label="Toggle details">
                {showInfo ? "▲ Hide" : "▼ Show"}
              </button>
            </div>

            {showInfo && (
              <div className="info-content">
                <p className="intro-text">
                  <strong>Alzheimer’s disease</strong> is a progressive neurodegenerative disorder caused by abnormal
                  accumulations of <em>beta-amyloid plaques</em> and <em>tau tangles</em>, leading to widespread neuronal cell
                  death and brain atrophy (shrinkage), primarily starting in the hippocampus.
                </p>

                <h3>MRI Diagnostic Classes</h3>
                <div className="stages-grid">
                  <div className="stage-card">
                    <h4>1. Non-Demented</h4>
                    <p>Preserved brain volume, intact hippocampus, and normal, narrow ventricles without marked atrophy.</p>
                  </div>

                  <div className="stage-card">
                    <h4>2. Very Mild Dementia</h4>
                    <p>Early structural decline localized to the entorhinal cortex and hippocampus with slight ventricle widening.</p>
                  </div>

                  <div className="stage-card">
                    <h4>3. Mild Dementia</h4>
                    <p>Evident atrophy in temporal lobes, expanding lateral ventricles, and pronounced deepening of cortical sulci.</p>
                  </div>

                  <div className="stage-card">
                    <h4>4. Moderate Dementia</h4>
                    <p>Severe, diffuse cerebral atrophy, massive ventricle enlargement, and extensive loss of cortical gray matter.</p>
                  </div>
                </div>
              </div>
            )}
          </section>

          {/* Sample Image & Text Box */}
          <div className="sample-combined-box">
            <p className="sample-info-text">
              ⚠️ Reference Guide: This box shows a sample image of an MRI scan.
            </p>
            <button 
              type="button"
              className="toggle-sample-btn"
              onClick={() => setShowSample(!showSample)}
            >
              {showSample ? "Hide Sample Image" : "Show Sample Image"}
            </button>
            
            {showSample && (
              <div className="sample-image-container">
                <img src={sampleMriImage} alt="Sample MRI Scan" className="sample-mri-img" />
              </div>
            )}
          </div>
        </>
      )}

      {/* Main layout wrapper for split screen */}
      <div className={`main-layout ${result ? "split-view" : ""}`}>
        
        {/* Left Column: Upload box / Image preview box */}
        <div className="left-column">
          {!preview ? (
            <div
              className={`drop ${dragging ? "active" : ""}`}
              onClick={() => inputRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragging(false);
                choose(e.dataTransfer.files[0]);
              }}
            >
              <p>
                <strong>Drag & drop</strong> an MRI image here, or click to browse
              </p>
              <small>JPG, PNG, BMP, TIFF, WEBP · max {MAX_MB} MB</small>
            </div>
          ) : (
            <div className="card">
              <img src={preview} alt="MRI preview" className="preview" />
              <div className="actions">
                <button onClick={analyze} disabled={loading} style={{ width: "100%" }}>
                  {loading ? "Analyzing…" : "Analyze MRI"}
                </button>
                <button className="secondary" onClick={reset} disabled={loading} style={{ width: "100%", marginTop: "10px" }}>
                  Remove
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Analysis Result Box appears here side-by-side */}
        {result && (
          <div className="right-column">
            <section className="card result sticky-result">
              <h2>{result.prediction}</h2>
              <p className="conf">
                Confidence: <strong>{(result.confidence * 100).toFixed(2)}%</strong>
              </p>
              {result.confidence < 0.6 && <div className="warn">Low confidence — interpret with extra caution.</div>}
              <h3>All class probabilities</h3>
              {sorted.map(([name, p]) => (
                <div key={name} className="row">
                  <div className="label">
                    <span>{name}</span>
                    <span>{(p * 100).toFixed(2)}%</span>
                  </div>
                  <div className="bar">
                    <div className={`fill ${name === result.prediction ? "top" : ""}`} style={{ width: `${p * 100}%` }} />
                  </div>
                </div>
              ))}
            </section>
          </div>
        )}

      </div>

      <input ref={inputRef} type="file" hidden accept="image/*" onChange={(e) => choose(e.target.files[0])} />

      {error && <div className="error">{error}</div>}

      <footer>For research/educational use only. Not a medical diagnosis.</footer>
    </main>
  );
}