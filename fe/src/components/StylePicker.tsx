import { useLayoutEffect, useState } from 'react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';

const styles = [
  { id: 'outfit', name: '01 · Outfit', description: 'Warm beige, bright red, bold type.', color: '#ede4dd' },
  { id: 'glass', name: '02 · Liquid Glass', description: 'Soft light, floating controls, frosted glass.', color: '#e7eef7' },
  { id: 'brutalist', name: '03 · Brutalist', description: 'Heavy lines, hard shadows, electric yellow.', color: '#f3f2ed' },
  { id: 'nocturne', name: '04 · Nocturne', description: 'A dark gallery with champagne details.', color: '#15181c' },
  { id: 'field', name: '05 · Field Notes', description: 'Sage, warm paper, and stitched edges.', color: '#f1eee4' },
  { id: 'minimal', name: '06 · Minimal', description: 'White, soft grey, charcoal. Just the essentials.', color: '#fafafa' },
  { id: 'original', name: 'Original · Thread', description: 'The original ivory and terracotta design.', color: '#f8f7f3' },
] as const;

type StyleId = typeof styles[number]['id'];
const storageKey = 'thread-visual-style';
const isStyle = (value: string | null): value is StyleId => styles.some(style => style.id === value);

function initialStyle(): StyleId {
  const requested = new URLSearchParams(window.location.search).get('style');
  if (isStyle(requested)) return requested;
  try {
    const saved = localStorage.getItem(storageKey);
    if (isStyle(saved)) return saved;
  } catch { /* The picker still works when local storage is unavailable. */ }
  return 'outfit';
}

export function StylePicker() {
  const [style, setStyle] = useState<StyleId>(initialStyle);
  const selected = styles.find(option => option.id === style)!;

  useLayoutEffect(() => {
    document.documentElement.dataset.style = style;
    try { localStorage.setItem(storageKey, style); } catch { /* Session-only fallback. */ }
  }, [style, selected.color]);

  function selectStyle(value: string) {
    if (!isStyle(value)) return;
    setStyle(value);
    // Keep a shareable style URL without navigating or resetting the demo.
    const url = new URL(window.location.href);
    url.searchParams.set('style', value);
    window.history.replaceState(null, '', url);
  }

  return (
    <Select value={style} onValueChange={selectStyle}>
    <SelectTrigger className="style-picker" title={selected.description} aria-label="Page style">
      <span className="style-swatch" aria-hidden="true" />
      <span className="style-picker-label">Style</span>
      <SelectValue>{selected.name}</SelectValue>
    </SelectTrigger>
    <SelectContent className="style-menu" align="end">
      {styles.map(option => <SelectItem key={option.id} value={option.id} textValue={option.name}><span className="style-option-copy"><span>{option.name}</span><small>{option.description}</small></span></SelectItem>)}
    </SelectContent>
    </Select>
  );
}
