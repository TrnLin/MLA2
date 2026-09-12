import type { Analysis, Metadata, SearchResult } from '../api/client';

export function ModelDetails({ metadata, analysis, search, imageUrl }: {
  metadata?: Metadata; analysis?: Analysis; search?: SearchResult; imageUrl: string | null;
}) {
  return <section className="advanced-section" aria-label="Advanced model information">
    <div className="advanced-heading"><div><div className="eyebrow">UNDER THE SURFACE</div><h2>Follow the thinking.</h2></div><span className="outline-tag">ADVANCED VIEW</span></div>
    <div className="pipeline">
      <div><span>INPUT</span><h3>Original photo</h3><p>JPG, PNG, or WebP</p></div>
      <span className="pipeline-arrow" aria-hidden="true">→</span>
      <div><span>PREPARE</span><h3>Each model’s settings</h3><p>RGB · keep shape · add padding</p></div>
      <span className="pipeline-arrow" aria-hidden="true">→</span>
      <div className="pipeline-branches">
        <div><span>TASKS 1–3 · 60 × 80</span><h3>Four models → Four labels</h3></div>
        <div><span>TASK 4 · 240 × 320</span><h3>128 image features → Similar items</h3></div>
      </div>
    </div>
    <p className="pipeline-note">Sizes are width × height in pixels. Tasks 1–3 use the original photo. Task 4 uses your chosen crop or the whole photo, then its saved resize and colour-scaling settings.</p>
    <div className="technical-grid">
      <details open className="technical-panel"><summary>Model configuration <span>VERIFIED PACKAGES</span></summary>
        <div className="technical-content"><table className="model-config-table"><caption className="visually-hidden">Loaded model settings</caption><thead><tr><th>Model</th><th>Input W × H</th><th>Parameters</th></tr></thead><tbody>
          {metadata ? Object.entries(metadata.models).map(([target, model]) => <tr key={target}><td title={model.run_id}>{model.name}<br /><small>{model.status === 'ready' ? 'Ready' : model.error || 'Unavailable'}</small></td><td>{model.width} × {model.height}</td><td>{model.parameters.toLocaleString()}</td></tr>) : <tr><td colSpan={3}>Waiting for the model server.</td></tr>}
        </tbody></table><div className="config-foot"><span>Compute device</span><span>CPU</span></div><div className="config-foot"><span>Image preparation</span><span>Saved settings · No random changes</span></div></div>
      </details>
      <details open className="technical-panel"><summary>Input & timing <span>THIS IMAGE</span></summary><div className="technical-content">
        <div className="preprocess-preview">{imageUrl && <img src={imageUrl} alt="Display preview of the original upload" />}<div><strong>Original image preview</strong><p>This display size does not set a model’s input size. Each model prepares its own copy on the server.</p></div></div>
        <div className="config-foot"><span>Four classifications</span><span>{analysis ? `${analysis.elapsed_ms.toFixed(1)} ms` : 'Waiting for image'}</span></div>
        {analysis && Object.entries(analysis.predictions).map(([target, prediction]) => <div className="config-foot" key={target}><span>{metadata?.models[target as keyof Metadata['models']]?.name || target}</span><span>{prediction.error ? 'Unavailable' : `${prediction.latency_ms.toFixed(1)} ms${prediction.cached ? ' (saved run)' : ''}`}</span></div>)}
        <div className="config-foot"><span>Similar-item search</span><span>{search ? `${search.latency_ms.toFixed(1)} ms${search.cached ? ' (saved run)' : ''}` : 'Waiting for search'}</span></div>
        <div className="config-foot"><span>Task 4 search area</span><span>{search ? search.crop ? `3:4 crop · left ${search.crop.left}, top ${search.crop.top}, right ${search.crop.right}, bottom ${search.crop.bottom}` : 'Whole image' : 'Choose a crop or the whole image'}</span></div>
        <p>Times are measured on the server. They exclude the browser’s upload time.</p>
        {analysis?.predictions.season?.review_required === null && <p>Season: no human-review threshold was saved with this model.</p>}
      </div></details>
    </div>
    <details className="technical-panel evaluation-panel"><summary>Saved evaluation & known limits <span>SEPARATE FROM THIS IMAGE</span></summary><div className="technical-content evaluation-content">
      {metadata?.evaluation.length ? metadata.evaluation.map(item => <div key={item.target}><h3>{item.title}</h3>{Object.entries(item.metrics).map(([name, value]) => <div className="config-foot" key={name}><span>{name}</span><span>{value}</span></div>)}<p>{item.note}</p><small>Source: {item.source}</small></div>) : <div><h3>Saved evaluation</h3><p>Saved scores are not available from the server yet.</p></div>}
      <div><h3>Keep the scores in context.</h3><p>Prediction scores are not accuracy. Season and occasion can be hard to see in a photo. Gender means the catalogue’s audience tag. Search scores measure image similarity, not a chance of being correct.</p></div>
    </div></details>
  </section>;
}
