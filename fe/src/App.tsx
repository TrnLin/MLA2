import { useEffect, useRef, useState, type ChangeEvent, type DragEvent } from 'react';
import { useSimilarItems } from './hooks/useSimilarItems';
import type { CropBox, SimilarItem, Target } from './api/client';
import { useAnalysis, useModelMetadata } from './hooks/useAnalysis';
import { ModelDetails } from './components/ModelDetails';
import { StylePicker } from './components/StylePicker';
import { ColorModeToggle } from './components/ColorModeToggle';
import { LayoutPicker } from './components/LayoutPicker';
import { CropEditor } from './components/CropEditor';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './components/ui/select';

type IconName = 'arrow' | 'upload' | 'grid' | 'scan' | 'chevron' | 'close' | 'shirt' | 'sun' | 'person' | 'bag' | 'code' | 'image' | 'check' | 'info';
function Icon({ name, size = 20 }: { name: IconName; size?: number }) {
  const paths: Record<IconName, React.ReactNode> = {
    arrow: <><path d="M5 12h14M13 6l6 6-6 6" /></>,
    upload: <><path d="M12 16V4m-5 5 5-5 5 5M4 16v4h16v-4" /></>,
    grid: <><rect x="3" y="3" width="7" height="7" rx="1" /><rect x="14" y="3" width="7" height="7" rx="1" /><rect x="3" y="14" width="7" height="7" rx="1" /><rect x="14" y="14" width="7" height="7" rx="1" /></>,
    scan: <><path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5M7 9h10M7 13h7M7 17h4" /></>,
    chevron: <path d="m9 5 7 7-7 7" />,
    close: <path d="m6 6 12 12M6 18 18 6" />,
    shirt: <path d="m8 3-6 4 3 5 3-2v11h8V10l3 2 3-5-6-4c-1 4-7 4-8 0Z" />,
    sun: <><circle cx="12" cy="12" r="4" /><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1 1m12 12 1 1M5 19l1-1M18 6l1-1" /></>,
    person: <><circle cx="12" cy="7" r="4" /><path d="M4 21v-3a8 8 0 0 1 16 0v3" /></>,
    bag: <><rect x="4" y="7" width="16" height="14" rx="2" /><path d="M8 9V6a4 4 0 0 1 8 0v3" /></>,
    code: <><path d="m7 7-5 5 5 5m10-10 5 5-5 5M14 3l-4 18" /></>,
    image: <><rect x="3" y="3" width="18" height="18" rx="2" /><circle cx="8" cy="8" r="1" /><path d="m3 17 6-6 4 4 3-3 5 5" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    info: <><circle cx="12" cy="12" r="9" /><path d="M12 11v6m0-10v1" /></>,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
}

type Sample = { id: string; image: string; name: string };
const samples: Sample[] = [
  { id: 'tshirt', image: '/samples/84.webp', name: 'T-shirt' },
  { id: 'watch', image: '/samples/93.webp', name: 'Watch' },
  { id: 'bag', image: '/samples/173.webp', name: 'Bag' },
  { id: 'dress', image: '/samples/177.webp', name: 'Dress' },
];


function App() {
  const [advanced, setAdvanced] = useState(false);
  const [selected, setSelected] = useState<Sample | null>(samples[0]);
  const [imageUrl, setImageUrl] = useState<string | null>(samples[0].image);
  const [filename, setFilename] = useState(`Sample 01 · ${samples[0].name}`);
  const [upload, setUpload] = useState<Blob | null>(null);
  const [count, setCount] = useState(5);
  const [error, setError] = useState('');
  const [dragging, setDragging] = useState(false);
  const [imageOptionsOpen, setImageOptionsOpen] = useState(true);
  const [activeMatch, setActiveMatch] = useState<SimilarItem | null>(null);
  const [searchSelection, setSearchSelection] = useState<{ source: string; crop: CropBox | null } | null>(null);
  const [cropEditorOpen, setCropEditorOpen] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const imageOptionsToggle = useRef<HTMLButtonElement>(null);
  const imageOptions = useRef<HTMLDivElement>(null);
  const generation = useRef(0);
  const objectUrl = useRef<string | null>(null);
  const timerIds = useRef<ReturnType<typeof setTimeout>[]>([]);
  const matchDialog = useRef<HTMLDialogElement>(null);
  const searchPanel = useRef<HTMLElement>(null);
  const searchFocus = useRef<'editor' | 'results' | null>(null);

  useEffect(() => () => {
    timerIds.current.forEach(clearTimeout);
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
  }, []);
  useEffect(() => { if (activeMatch !== null) matchDialog.current?.showModal(); else matchDialog.current?.close(); }, [activeMatch]);

  const metadata = useModelMetadata();
  const analysis = useAnalysis(imageUrl, upload, metadata.data?.version);
  const selection = searchSelection?.source === imageUrl ? searchSelection : null;
  const isChoosingCrop = Boolean(imageUrl && (cropEditorOpen || !selection));
  const similarItems = useSimilarItems(
    selection && !isChoosingCrop ? analysis.data?.image_id ?? null : null,
    count, analysis.data?.version, selection?.crop ?? null,
  );
  const requestError = analysis.error || metadata.error;
  const degraded = metadata.data && Object.values(metadata.data.models).some(model => model.status !== 'ready');
  const status = !imageUrl ? 'empty' : requestError ? 'error' : analysis.data ? 'ready' : 'loading';
  useEffect(() => { setActiveMatch(null); }, [imageUrl, count, searchSelection, cropEditorOpen]);
  useEffect(() => {
    if (!searchFocus.current) return;
    const selector = searchFocus.current === 'editor' ? '.crop-editor' : '.crop-search-summary button';
    searchPanel.current?.querySelector<HTMLElement>(selector)?.focus({ preventScroll: true });
    searchFocus.current = null;
  }, [isChoosingCrop, searchSelection]);
  const applySearchCrop = (crop: CropBox | null) => {
    if (!imageUrl) return;
    searchFocus.current = 'results';
    setSearchSelection({ source: imageUrl, crop });
    setCropEditorOpen(false);
  };
  const editSearchCrop = () => { searchFocus.current = 'editor'; setCropEditorOpen(true); };
  const cancelSearchCrop = () => { searchFocus.current = 'results'; setCropEditorOpen(false); };
  const retryAnalysis = () => {
    if (metadata.isError) void metadata.refetch();
    else analysis.retry();
  };

  const later = (callback: () => void, ms: number) => { timerIds.current.push(setTimeout(callback, ms)); };
  const collapseImageOptions = () => {
    // Keep keyboard focus on a visible control when its picker closes.
    if (imageOptions.current?.contains(document.activeElement)) {
      imageOptionsToggle.current?.focus({ preventScroll: true });
    }
    setImageOptionsOpen(false);
  };
  const chooseSample = (sample: Sample, index: number) => {
    generation.current += 1;
    setSearchSelection(null); setCropEditorOpen(false);
    setUpload(null);
    setSelected(sample); setImageUrl(sample.image); setFilename(`Sample 0${index + 1} · ${sample.name}`);
    setError('');
    collapseImageOptions();
  };
  const clearImage = () => {
    setSearchSelection(null); setCropEditorOpen(false);
    generation.current += 1; setImageUrl(null); setUpload(null); setSelected(null); setError('');
    setImageOptionsOpen(true);
    imageOptionsToggle.current?.focus({ preventScroll: true });
    if (fileInput.current) fileInput.current.value = '';
  };
  const readFile = (file: File | undefined) => {
    if (!file) return;
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) { setError('Choose a JPG, PNG, or WebP image.'); return; }
    if (file.size > 10 * 1024 * 1024) { setError('This image is too large. Choose an image under 10 MB.'); return; }
    const current = ++generation.current;
    const url = URL.createObjectURL(file);
    const checkImage = new Image();
    checkImage.onload = () => {
      if (current !== generation.current) { URL.revokeObjectURL(url); return; }
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
      objectUrl.current = url;
      setSearchSelection(null); setCropEditorOpen(false);
      setImageUrl(url); setUpload(file); setFilename(file.name); setSelected(null); setError('');
      collapseImageOptions();
    };
    checkImage.onerror = () => { URL.revokeObjectURL(url); if (current === generation.current) setError('We could not open this image. Please choose another.'); };
    checkImage.src = url;
  };
  const onUpload = (event: ChangeEvent<HTMLInputElement>) => { readFile(event.target.files?.[0]); event.target.value = ''; };
  const onDrop = (event: DragEvent) => { event.preventDefault(); setDragging(false); readFile(event.dataTransfer.files[0]); };
  const definitions: { target: Target; task: string; title: string; icon: IconName; description: string }[] = [
    { target: 'articleType', task: '01', title: 'Item type', icon: 'shirt', description: 'The shape of your style.' },
    { target: 'season', task: '02', title: 'Season', icon: 'sun', description: 'A season to wear it in.' },
    { target: 'gender', task: '03', title: 'Gender category', icon: 'person', description: 'The catalogue category.' },
    { target: 'usage', task: '03', title: 'Occasion', icon: 'bag', description: 'Where it fits into your day.' },
  ];
  const cards = definitions.map(card => {
    const prediction = analysis.data?.predictions[card.target];
    const scores = Object.entries(prediction?.probabilities ?? {}).sort((a, b) => b[1] - a[1]);
    return { ...card, prediction, scores: card.target === 'articleType' ? scores.slice(0, 5) : scores };
  });

  return (
    <>
      <a className="skip-link" href="#workspace">Skip to demo</a>
      <header className="site-header compact-header">
        <a className="wordmark" href="/" aria-label="Thread home"><span className="brand-symbol" aria-hidden="true"><i /><i /><i /></span>thread<span className="brand-period">.</span></a>
        <span className="header-caption">FASHION INTELLIGENCE</span>
        <div className="header-actions"><StylePicker /><ColorModeToggle /><a className="about-button" href="/" aria-label="About the project"><span className="about-label">About the project</span><Icon name="arrow" size={16} /></a></div>
      </header>

      <main className={advanced ? 'is-advanced' : undefined}>
        <section className="demo-toolbar" aria-labelledby="demo-title">
          <div className="demo-toolbar-copy"><h1 id="demo-title">Explore a piece.</h1><p>One image. Four tasks. A closer look.</p></div>
          <div className="toolbar-controls"><LayoutPicker /><div className="mode-switch" role="group" aria-label="Display mode"><button aria-pressed={!advanced} className={!advanced ? 'active' : ''} onClick={() => setAdvanced(false)}>Standard</button><button aria-pressed={advanced} className={advanced ? 'active' : ''} onClick={() => setAdvanced(true)}><Icon name="code" size={16} /> Advanced</button></div></div>
        </section>

        <div className="demo-notice compact-notice"><span className="demo-dot" /><strong>{metadata.isError ? 'Server offline' : degraded ? 'Some models unavailable' : metadata.data ? 'Live models' : 'Connecting'}</strong><span>{metadata.isError ? 'Start the local model server, then try again.' : metadata.data ? 'Real predictions · Local model server' : 'Connecting to the local model server…'}</span></div>

        <div className="demo-body">
        <section id="workspace" className="workspace" aria-label="Fashion demo">
          <aside className="upload-column">
            <div className="section-heading"><h2><span className="step">01</span> Start with a piece</h2><span className="small-label">YOUR IMAGE</span></div>
            <div className={`upload-frame ${dragging ? 'dragging' : ''} ${imageUrl ? 'has-image' : 'no-image'}`} onDragOver={event => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={onDrop}>
              {imageUrl ? <><div className="image-topline"><span><span className="status-dot" />{selected ? 'SAMPLE IMAGE' : 'YOUR UPLOAD'}</span><button className="icon-button" aria-label="Clear image" onClick={clearImage}><Icon name="close" size={17} /></button></div><img className="query-image" src={imageUrl} alt={selected?.name || 'Your uploaded fashion item'} /><span className="photo-corner corner-tl" /><span className="photo-corner corner-br" /><div className="image-caption"><span>{selected ? 'FROM THE SAMPLE COLLECTION' : 'PROCESSED ON THIS COMPUTER'}</span><span>↗</span></div></> : <button className="empty-upload" onClick={() => fileInput.current?.click()}><span className="upload-circle"><Icon name="upload" size={27} /></span><strong>A new discovery starts here.</strong><span>Drop a photo or click to browse</span><small>JPG, PNG or WebP · Up to 10 MB</small></button>}
              {dragging && <div className="drop-overlay"><Icon name="upload" size={32} /><strong>Drop your image here</strong></div>}
            </div>
            {imageUrl && <div className="filename" title={filename}><Icon name="image" size={15} /><span>{filename}</span></div>}
            <input ref={fileInput} type="file" accept="image/jpeg,image/png,image/webp" onChange={onUpload} className="visually-hidden" tabIndex={-1} aria-label="Upload a fashion photo" />
            <button ref={imageOptionsToggle} className="upload-button image-options-toggle" aria-expanded={imageOptionsOpen} aria-controls="image-options" onClick={() => setImageOptionsOpen(open => !open)}><Icon name="image" size={18} />{imageUrl ? 'Change image' : 'Image options'}<span><Icon name="chevron" size={16} /></span></button>
            <div ref={imageOptions} id="image-options" className="image-options" hidden={!imageOptionsOpen}>
            <button className="upload-button" onClick={() => fileInput.current?.click()}><Icon name="upload" size={18} />{imageUrl ? 'Upload your own image' : 'Choose an image'}<span>↗</span></button>
            <p className="upload-help">Or drag a photo into the frame · Max 10 MB</p>
            <div className="sample-heading"><span>Just looking? Try a sample.</span><span>01 — 04</span></div>
            <div className="sample-list">{samples.map((sample, index) => <button key={sample.id} aria-label={`Try ${sample.name}`} aria-pressed={selected?.id === sample.id} className={`sample-button ${selected?.id === sample.id ? 'selected' : ''}`} onClick={() => chooseSample(sample, index)}><img src={sample.image} alt="" /><span>0{index + 1}</span>{selected?.id === sample.id && <i><Icon name="check" size={10} /></i>}</button>)}</div>
            </div>
            {error && <div className="error-message" role="alert"><Icon name="info" size={18} /><span>{error}</span></div>}
            <p className="privacy-note">Your photo is sent to the local model server and kept in its cache on this computer.</p>
          </aside>

          <div className="results-column">
            <div className="section-heading"><h2><span className="step">02</span> A closer look</h2><span className="result-status" aria-live="polite"><span className={status === 'loading' ? 'status-dot pulsing' : 'status-dot'} />{status === 'loading' ? 'READING YOUR IMAGE' : status === 'ready' ? 'MODEL RESULTS' : status === 'error' ? 'COULD NOT LOAD RESULTS' : 'WAITING FOR AN IMAGE'}</span></div>

            {status === 'empty' ? <div className="result-empty"><Icon name="scan" size={38} /><h3>One image. A new perspective.</h3><p>Add a photo to explore its details<br />and discover similar pieces.</p><button className="text-button" onClick={() => fileInput.current?.click()}>Choose your first image <Icon name="arrow" size={17} /></button></div> : status === 'error' ? <div className="result-empty" role="alert"><Icon name="info" size={38} /><h3>Could not load predictions.</h3><p>{requestError?.message}</p><button className="primary-button" onClick={retryAnalysis}>Try again <Icon name="arrow" size={18} /></button></div> : status === 'loading' ? <div className="loading-results" role="status"><div className="loading-title"><span className="spinner" /><span>Reading your image…</span></div><div className="detail-grid">{[1, 2, 3, 4].map(i => <div className="detail-card skeleton" key={i}><span /><strong /><span /></div>)}</div><p>Each model is using its own saved image settings.</p></div> : <section id="details-panel" aria-label="Item details" className="tab-panel">
              <div className="results-intro"><h3>The details make the difference.</h3><p>Four ways to get to know this piece.</p></div>
              <div className="detail-grid">{cards.map((card, index) => <article className="detail-card" key={card.title} style={{ animationDelay: `${index * 65}ms` }}><div className="card-top"><span>{card.title}</span><Icon name={card.icon} size={21} /></div><strong>{card.prediction?.label ?? 'Unavailable'}</strong>{card.prediction?.error && <p role="alert">{card.prediction.error}</p>}<div className="card-bottom"><span>{card.description}</span><span className="task-label">TASK {card.task}</span></div>{advanced && <div className="score-bars"><div className="score-heading">MODEL SCORES</div>{card.scores.map(([label, score], i) => <div className="score-row" key={`${label}-${i}`}><span>{label}</span><span>{(score * 100).toFixed(1)}%</span><div className="score-track"><i style={{ width: `${(score * 100).toFixed(1)}%` }} /></div></div>)}</div>}</article>)}</div>
              {cards.some(card => card.prediction?.error) && <button className="text-button" onClick={() => analysis.retry()}>Retry predictions <Icon name="arrow" size={16} /></button>}
              <p className="prediction-note"><Icon name="info" size={15} /><span>Model scores are suggestions, not certainty. Gender refers to product categories.</span></p>
            </section>}
            {imageUrl && <section ref={searchPanel} className="similar-banner" aria-labelledby="similar-heading">
              <div className="similar-banner-heading"><div><span className="eyebrow"><Icon name="grid" size={15} /> TASK 04 · VISUAL SEARCH</span><h3 id="similar-heading">Familiar feel. Fresh finds.</h3><p>A few more pieces to explore.</p></div><div className="count-select"><span>Show</span><Select value={String(count)} onValueChange={value => setCount(Number(value))}><SelectTrigger aria-label="Results to show"><SelectValue /></SelectTrigger><SelectContent align="end"><SelectItem value="5">5 items</SelectItem><SelectItem value="10">10 items</SelectItem></SelectContent></Select></div></div>
              {isChoosingCrop ? <CropEditor key={imageUrl} imageUrl={imageUrl}
                initialCrop={selection?.crop} onApply={applySearchCrop}
                onWholeImage={() => applySearchCrop(null)}
                onCancel={selection ? cancelSearchCrop : undefined} /> : <>
              <div className="crop-search-summary"><span>{selection?.crop ? '3:4 search crop' : 'Whole image'}</span><div>
                <button type="button" onClick={editSearchCrop}>Adjust crop</button>
                {selection?.crop && <button type="button" onClick={() => applySearchCrop(null)}>Use whole image</button>}
              </div></div>
              {similarItems.isError ? <div className="search-empty" role="alert"><Icon name="info" size={30} /><h3>Could not load similar items.</h3><p>{similarItems.error?.message || 'Please try the search again.'}</p><button className="primary-button" onClick={() => void similarItems.refetch()}>Try again <Icon name="arrow" size={18} /></button></div> : status === 'error' ? <p className="prediction-note">Retry the image above to start its search.</p> : status !== 'ready' || similarItems.isPending ? <div role="status"><div className="loading-title"><span className="spinner" />Finding similar pieces…</div><div className="matches-grid">{Array.from({ length: count }, (_, i) => <div className="match-skeleton" key={i} />)}</div></div> : <><div className="match-notice"><span>{similarItems.data?.items.length ?? 0} matches</span><span>{selection?.crop ? 'MATCHED BY YOUR CROP' : 'MATCHED BY YOUR IMAGE'}</span></div><div className="matches-grid">{(similarItems.data?.items ?? []).map((item, index) => <button className="match-card" key={item.image} onClick={() => setActiveMatch(item)}><div className="match-image"><img src={item.image} alt={item.name} loading="lazy" /><span className="match-rank">{String(index + 1).padStart(2, '0')}</span><span className="match-expand"><Icon name="arrow" size={15} /></span></div><span className="match-name">{item.name}</span>{advanced ? <span className="match-score">Cosine similarity <b>{item.score.toFixed(2)}</b></span> : <span className="match-subtitle">Product collection</span>}</button>)}</div><p className="prediction-note"><Icon name="info" size={15} /><span>Ranked by image similarity. Similarity scores are not prediction probabilities.</span></p>{advanced && <div className="search-config"><span><b>Search method</b>{similarItems.data?.method ?? 'R5'}</span><span><b>Gallery split</b>{similarItems.data?.gallery_size.toLocaleString()} development products</span><span><b>Results requested</b>{count}</span><span><b>Actual search time</b>{similarItems.data?.latency_ms.toFixed(1)} ms{similarItems.data?.cached ? ' (saved run)' : ''}</span></div>}</>}
              </>}
            </section>}
          </div>
        </section>

        {advanced && <ModelDetails metadata={metadata.data} analysis={analysis.data} search={similarItems.data} imageUrl={imageUrl} />}
        </div>

        <section className="how-it-works" aria-label="How the demo works"><div className="how-title"><span className="eyebrow">ONE IMAGE. FOUR TASKS.</span><h2>Small details.<br /><em>Bigger picture.</em></h2></div><div className="how-step"><span>01 — 03</span><h3>Understand the piece</h3><p>Item, season, gender category, occasion.<br />The essentials, all in one place.</p></div><div className="how-step"><span>04</span><h3>Find its kind</h3><p>Look beyond a label.<br />Explore visually similar pieces.</p></div><button className="how-link" onClick={() => { setAdvanced(!advanced); if (!advanced) later(() => document.querySelector('.advanced-section')?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50); }}>{advanced ? 'Back to the essentials' : 'Curious how it works?'}<Icon name="arrow" size={20} /></button></section>
      </main>
      <footer><a className="footer-wordmark" href="#">thread.</a><span>FASHION INTELLIGENCE · COSC2753</span><span>Made to look a little closer.</span><span className="footer-preview">LOCAL MODEL DEMO</span></footer>

      <dialog ref={matchDialog} onCancel={() => setActiveMatch(null)} onClick={event => { if (event.target === event.currentTarget) setActiveMatch(null); }} aria-labelledby="match-title" className="match-dialog"><button className="dialog-close icon-button" onClick={() => setActiveMatch(null)} aria-label="Close item preview"><Icon name="close" /></button>{activeMatch !== null && <><img src={activeMatch.image} alt={activeMatch.name} /><div className="eyebrow">SIMILAR ITEM / {String(activeMatch.rank).padStart(2, '0')}</div><h2 id="match-title">{activeMatch.name}</h2><p>A product found by the visual-search model in the saved development gallery.</p>{advanced && <p>Cosine similarity: <strong>{activeMatch.score.toFixed(2)}</strong></p>}</>}</dialog>
    </>
  );
}

export default App;
