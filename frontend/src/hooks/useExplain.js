import { useMutation } from '@tanstack/react-query';
import { explainEyeDisease } from '@/services/predictionApi';

export function useExplain() {
  return useMutation({
    mutationKey: ['explain-eye-disease'],
    mutationFn: ({ file, signal } = {}) => explainEyeDisease(file, { signal }),
    retry: false,
  });
}
