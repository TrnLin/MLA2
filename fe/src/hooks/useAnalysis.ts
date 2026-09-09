import { useEffect, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { analyzeImage, getMetadata, imageIdentity, ModelVersionChanged, requireVersion } from '../api/client';

type PreparedImage = { source: string; blob: Blob; hash: string };

export function useModelMetadata() {
  return useQuery({
    queryKey: ['model-metadata'], queryFn: ({ signal }) => getMetadata(signal),
    staleTime: 30_000, refetchInterval: 30_000, retry: false,
  });
}

export function useAnalysis(imageUrl: string | null, upload: Blob | null, version?: string) {
  const queryClient = useQueryClient();
  const [prepared, setPrepared] = useState<PreparedImage | null>(null);
  const [prepareError, setPrepareError] = useState<{ source: string; error: Error } | null>(null);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    if (!imageUrl) return;
    const controller = new AbortController();
    const prepare = async () => {
      let blob = upload;
      if (!blob) {
        const response = await fetch(imageUrl, { signal: controller.signal });
        if (!response.ok) throw new Error('Could not open this sample image. Try another image.');
        blob = await response.blob();
      }
      const hash = await imageIdentity(blob);
      if (!controller.signal.aborted) {
        setPrepared({ source: imageUrl, blob, hash });
        setPrepareError(null);
      }
    };
    void prepare().catch(error => {
      if (!controller.signal.aborted) setPrepareError({ source: imageUrl, error });
    });
    return () => controller.abort();
  }, [imageUrl, upload, attempt]);
  // Never allow the previous image's data to appear during decode/hash work.
  const current = prepared?.source === imageUrl ? prepared : null;
  const query = useQuery({
    queryKey: ['analysis', version, current?.hash],
    queryFn: async ({ signal }) => {
      const result = await analyzeImage(current!.blob, signal);
      try { return requireVersion(result, version!); }
      catch (error) {
        void queryClient.invalidateQueries({ queryKey: ['model-metadata'] });
        throw error;
      }
    },
    enabled: Boolean(current && version), staleTime: 5 * 60_000,
    gcTime: 5 * 60_000, retry: false,
  });
  const error = prepareError?.source === imageUrl ? prepareError.error : query.error instanceof ModelVersionChanged ? null : query.error;
  return {
    ...query,
    data: current && version ? query.data : undefined,
    error,
    isError: Boolean(error),
    retry: () => {
      if (prepareError?.source === imageUrl) { setPrepareError(null); setAttempt(n => n + 1); }
      else void query.refetch();
    },
  };
}
