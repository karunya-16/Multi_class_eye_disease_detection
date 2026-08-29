import { useMutation, useQuery } from '@tanstack/react-query';
import {
  fetchAnalysisDetail,
  fetchAnalysisHistory,
  saveAnalysisResult,
} from '@/services/historyApi';

export function useAnalysisHistory({
  predictedLabel = '',
  query = '',
  startDate = '',
  endDate = '',
  page = 1,
  pageSize = 8,
  enabled = true,
} = {}) {
  return useQuery({
    queryKey: ['analysis-history', predictedLabel, query, startDate, endDate, page, pageSize],
    queryFn: ({ signal }) =>
      fetchAnalysisHistory({
        predictedLabel: predictedLabel || undefined,
        query: query || undefined,
        startDate: startDate || undefined,
        endDate: endDate || undefined,
        page,
        pageSize,
        signal,
      }),
    enabled,
    retry: false,
  });
}

export function useAnalysisDetail(id) {
  const analysisId = Number(id);
  return useQuery({
    queryKey: ['analysis-history', 'detail', analysisId],
    queryFn: ({ signal }) => fetchAnalysisDetail(analysisId, { signal }),
    enabled: Number.isFinite(analysisId) && analysisId > 0,
    retry: false,
  });
}

export function useSaveAnalysis() {
  return useMutation({
    mutationKey: ['save-analysis-history'],
    mutationFn: (payload) => saveAnalysisResult(payload),
    retry: false,
  });
}
