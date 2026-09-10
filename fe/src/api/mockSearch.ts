export type SimilarItem = {
  id: number;
  name: string;
  image: string;
  score: number;
  rank: number;
};

const gallery: SimilarItem[] = [
  { id: 83, name: 'Blue & Black Check Shirt' }, { id: 85, name: 'Man Plaid Shirt' },
  { id: 87, name: 'Check Shirt' }, { id: 84, name: 'Gigabyte Aorus T-Shirt' },
  { id: 86, name: 'Short Sleeve Shirt' }, { id: 90, name: 'Puma Future Rider Trainers' },
  { id: 173, name: 'Heshe Women’s Leather Bag' }, { id: 177, name: 'Black Women’s Gown' },
  { id: 175, name: 'White Faux Leather Backpack' }, { id: 92, name: 'Sports Sneakers Off White & Red' },
].map((item, index) => ({
  ...item,
  image: `/samples/${item.id}.webp`,
  score: [0.94, 0.91, 0.89, 0.86, 0.84, 0.82, 0.80, 0.78, 0.76, 0.74][index],
  rank: index + 1,
}));

// Local fixture only. Replace this function with the FastAPI request later.
export async function getMockSimilarItems(imageKey: string, limit: number, signal: AbortSignal) {
  if (!imageKey) throw new Error('Choose an image before searching.');
  signal.throwIfAborted();
  await new Promise<void>((resolve, reject) => {
    const abort = () => {
      clearTimeout(timer);
      reject(signal.reason);
    };
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', abort);
      resolve();
    }, 1100);
    signal.addEventListener('abort', abort, { once: true });
  });
  return gallery.slice(0, limit);
}
