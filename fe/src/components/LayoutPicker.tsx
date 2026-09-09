import { useLayoutEffect, useState } from 'react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';

type Layout = 'current' | 'studio';
const storageKey = 'thread-demo-layout';
const isLayout = (value: string | null): value is Layout => value === 'current' || value === 'studio';

function initialLayout(): Layout {
  const requested = new URLSearchParams(window.location.search).get('layout');
  if (isLayout(requested)) return requested;
  try {
    const saved = localStorage.getItem(storageKey);
    if (isLayout(saved)) return saved;
  } catch { /* Keep the switch usable without storage. */ }
  return 'studio';
}

export function LayoutPicker() {
  const [layout, setLayout] = useState<Layout>(initialLayout);
  useLayoutEffect(() => {
    document.documentElement.dataset.layout = layout;
    try { localStorage.setItem(storageKey, layout); } catch { /* Session-only fallback. */ }
  }, [layout]);

  return <div className="layout-picker">
    <span>Layout</span>
    <Select value={layout} onValueChange={value => {
      if (!isLayout(value)) return;
      setLayout(value);
      const url = new URL(window.location.href);
      url.searchParams.set('layout', value);
      window.history.replaceState(null, '', url);
    }}>
      <SelectTrigger aria-label="Page layout"><SelectValue /></SelectTrigger>
      <SelectContent><SelectItem value="current">Current</SelectItem><SelectItem value="studio">Studio · New</SelectItem></SelectContent>
    </Select>
  </div>;
}
