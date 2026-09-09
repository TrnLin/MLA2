import { useQuery, useQueryClient } from '@tanstack/react-query';
import { getSimilarItems, ModelVersionChanged, requireVersion, searchQueryKey, type CropBox } from '../api/client';

export function useSimilarItems(imageKey: string | null, limit: number, version?: string, crop: CropBox | null = null) {
  const queryClient = useQueryClient();
  const query = useQuery({
    queryKey: searchQueryKey(version, imageKey, limit, crop),
    queryFn: async ({ signal }) => {
      const result = await getSimilarItems(imageKey!, limit, signal, crop);
      try { return requireVersion(result, version!); }
      catch (error) {
        void queryClient.invalidateQueries({ queryKey: ['model-metadata'] });
        throw error;
      }
    },
    enabled: Boolean(version && imageKey),
    staleTime: 5 * 60 * 1000,
    gcTime: 5 * 60 * 1000,
    retry: false,
  });
  const changed = query.error instanceof ModelVersionChanged;
  return { ...query, data: imageKey && version ? query.data : undefined,
    isError: query.isError && !changed, isPending: query.isPending || changed };
}
