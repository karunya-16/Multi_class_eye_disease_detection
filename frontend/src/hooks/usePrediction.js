import { useMutation } from '@tanstack/react-query';
import { predictEyeDisease } from '@/services/predictionApi';

export function usePrediction() {
  return useMutation({
    mutationKey: ['predict-eye-disease'],
    mutationFn: ({ file, signal } = {}) => predictEyeDisease(file, { signal }),
    retry: false,
  });
}
