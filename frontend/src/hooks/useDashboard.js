import { useQuery } from '@tanstack/react-query';
import { fetchDashboardAnalytics } from '@/services/dashboardApi';

export function useDashboardAnalytics({
  predictedLabel = '',
  startDate = '',
  endDate = '',
  enabled = true,
} = {}) {
  return useQuery({
    queryKey: ['dashboard', predictedLabel, startDate, endDate],
    queryFn: ({ signal }) =>
      fetchDashboardAnalytics({
        predictedLabel: predictedLabel || undefined,
        startDate: startDate || undefined,
        endDate: endDate || undefined,
        signal,
      }),
    enabled,
    retry: false,
  });
}
