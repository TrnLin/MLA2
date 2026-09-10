import { useEffect, useLayoutEffect, useState } from 'react';

type ColorMode = 'light' | 'dark';
const storageKey = 'thread-color-mode';
const isMode = (value: string | null): value is ColorMode => value === 'light' || value === 'dark';

function initialPreference(): ColorMode | null {
  const requested = new URLSearchParams(window.location.search).get('appearance');
  if (isMode(requested)) return requested;
  try {
    const saved = localStorage.getItem(storageKey);
    if (isMode(saved)) return saved;
  } catch { /* Use the system setting when storage is unavailable. */ }
  return null;
}

export function ColorModeToggle() {
  const [preference, setPreference] = useState<ColorMode | null>(initialPreference);
  const [systemDark, setSystemDark] = useState(() => window.matchMedia('(prefers-color-scheme: dark)').matches);
  const mode = preference ?? (systemDark ? 'dark' : 'light');

  useEffect(() => {
    if (preference !== null) return;
    const media = window.matchMedia('(prefers-color-scheme: dark)');
    const update = () => setSystemDark(media.matches);
    update();
    media.addEventListener('change', update);
    return () => media.removeEventListener('change', update);
  }, [preference]);

  useLayoutEffect(() => {
    const root = document.documentElement;
    root.dataset.appearance = mode;
    if (preference !== null) {
      try { localStorage.setItem(storageKey, preference); } catch { /* Session-only fallback. */ }
    }
    const updateBrowserColor = () => {
      const styles = getComputedStyle(root);
      document.querySelector('meta[name="theme-color"]')?.setAttribute('content', styles.getPropertyValue('--page').trim() || styles.backgroundColor);
    };
    updateBrowserColor();
    const observer = new MutationObserver(updateBrowserColor);
    observer.observe(root, { attributes: true, attributeFilter: ['data-style'] });
    return () => observer.disconnect();
  }, [mode, preference]);

  return <button className="color-mode-toggle" aria-label={`Switch to ${mode === 'dark' ? 'light' : 'dark'} mode`} title={`Switch to ${mode === 'dark' ? 'light' : 'dark'} mode`} onClick={() => {
    const next = mode === 'dark' ? 'light' : 'dark';
    setPreference(next);
    const url = new URL(window.location.href);
    url.searchParams.set('appearance', next);
    window.history.replaceState(null, '', url);
  }}>
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {mode === 'dark' ? <><circle cx="12" cy="12" r="4" /><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1 1m12 12 1 1M5 19l1-1M18 6l1-1" /></> : <path d="M20.8 13.1A9 9 0 0 1 10.9 3.2a9 9 0 1 0 9.9 9.9Z" />}
    </svg>
  </button>;
}
