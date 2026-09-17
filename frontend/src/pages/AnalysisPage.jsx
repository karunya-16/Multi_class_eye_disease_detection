import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { toast } from 'sonner';
import { LoaderCircle, RotateCcw } from 'lucide-react';
import { motion } from 'motion/react';
import { Card } from '@/components/ui/card';
import ImageUploader from '@/components/ImageUploader';
import ResultPanel from '@/components/ResultPanel';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { useExplain } from '@/hooks/useExplain';
import { usePrediction } from '@/hooks/usePrediction';
import { GRADCAM_LOADING_MESSAGE, LOADING_MESSAGES, MODEL_SCOPE_NOTE } from '@/lib/constants';
import { analysisFormSchema } from '@/lib/validation';
import { PredictionApiError } from '@/services/predictionApi';

const ANALYZE_MESSAGES = [...LOADING_MESSAGES, GRADCAM_LOADING_MESSAGE];

function createRequestId() {
  if (globalThis.crypto?.randomUUID) {
    return globalThis.crypto.randomUUID();
  }
  return `analysis_${Date.now()}_${Math.random().toString(16).slice(2, 10)}`;
}

export default function AnalysisPage() {
  const [previewUrl, setPreviewUrl] = useState('');
  const [inlineError, setInlineError] = useState('');
  const [gradcamError, setGradcamError] = useState('');
  const [loadingIndex, setLoadingIndex] = useState(0);
  const [historyId, setHistoryId] = useState(null);
  const abortRef = useRef(null);
  const requestIdRef = useRef(createRequestId());
  const queryClient = useQueryClient();
  const explain = useExplain();
  const prediction = usePrediction();

  function abortInFlight() {
    abortRef.current?.abort();
    abortRef.current = null;
  }

  function beginNewRequest() {
    requestIdRef.current = createRequestId();
    setHistoryId(null);
  }

  const form = useForm({
    resolver: zodResolver(analysisFormSchema),
    defaultValues: { file: undefined },
    mode: 'onSubmit',
  });

  const file = form.watch('file');
  const result = explain.data || prediction.data;
  const requestPending = explain.isPending || prediction.isPending;
  const analyzingFirstPass = requestPending && !result;
  const retryingGradCam = explain.isPending && Boolean(result);

  useEffect(() => {
    if (!file) {
      setPreviewUrl((current) => {
        if (current) URL.revokeObjectURL(current);
        return '';
      });
      return undefined;
    }
    const nextUrl = URL.createObjectURL(file);
    setPreviewUrl((current) => {
      if (current) URL.revokeObjectURL(current);
      return nextUrl;
    });
    return () => URL.revokeObjectURL(nextUrl);
  }, [file]);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    if (!analyzingFirstPass) {
      setLoadingIndex(0);
      return undefined;
    }
    const timer = window.setInterval(() => {
      setLoadingIndex((index) => (index + 1) % ANALYZE_MESSAGES.length);
    }, 1400);
    return () => window.clearInterval(timer);
  }, [analyzingFirstPass]);

  function clearExplainability() {
    setGradcamError('');
    explain.reset();
  }

  function handleSelect(nextFile) {
    setInlineError('');
    abortInFlight();
    clearExplainability();
    prediction.reset();
    beginNewRequest();
    form.setValue('file', nextFile, { shouldValidate: true, shouldDirty: true });
  }

  function handleRemove() {
    setInlineError('');
    abortInFlight();
    clearExplainability();
    prediction.reset();
    beginNewRequest();
    form.reset({ file: undefined });
  }

  function handleNewAnalysis() {
    setInlineError('');
    abortInFlight();
    clearExplainability();
    prediction.reset();
    beginNewRequest();
    form.reset({ file: undefined });
  }

  function handleUploadError(message) {
    setInlineError(message);
    toast.error(message);
  }

  function rememberSaved(result) {
    const savedId = Number(result?.history_id);
    if (!Number.isFinite(savedId) || savedId <= 0) {
      toast.error('The result is available, but it could not be saved to history.');
      return null;
    }
    setHistoryId(savedId);
    queryClient.invalidateQueries({ queryKey: ['analysis-history'] });
    queryClient.invalidateQueries({ queryKey: ['dashboard'] });
    return savedId;
  }

  const onSubmit = form.handleSubmit(async (values) => {
    if (requestPending) return;
    setInlineError('');
    setGradcamError('');
    abortInFlight();
    explain.reset();
    prediction.reset();
    beginNewRequest();
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const explained = await explain.mutateAsync({
        file: values.file,
        signal: controller.signal,
      });
      toast.success(
        `${explained.disease} · Model Confidence ${Number(explained.confidence_percentage).toFixed(2)}%`,
      );
      rememberSaved(explained);
      if (explained.gradcam_available !== true) {
        const message =
          explained.gradcam_error ||
          'The prediction succeeded, but Grad-CAM could not be generated.';
        setGradcamError(message);
        toast.error(message);
      }
    } catch (error) {
      if (error?.name === 'AbortError') return;
      try {
        const predicted = await prediction.mutateAsync({
          file: values.file,
          signal: controller.signal,
        });
        toast.success(
          `${predicted.disease} · Model Confidence ${Number(predicted.confidence_percentage).toFixed(2)}%`,
        );
        rememberSaved(predicted);
        const message =
          'The prediction succeeded, but Grad-CAM could not be generated for this image.';
        setGradcamError(message);
        toast.error(message);
      } catch (predictError) {
        if (predictError?.name === 'AbortError') return;
        const message =
          predictError instanceof PredictionApiError || error instanceof PredictionApiError
            ? (predictError instanceof PredictionApiError ? predictError.message : error.message)
            : 'The analysis could not be completed. Please try again.';
        setInlineError(message);
        toast.error(message);
      }
    }
  });

  async function handleRetryGradCam() {
    if (!file || requestPending) return;
    setGradcamError('');
    abortInFlight();
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const explained = await explain.mutateAsync({
        file,
        signal: controller.signal,
      });
      rememberSaved(explained);
      if (explained.gradcam_available === true) {
        toast.success('Grad-CAM generated');
      } else {
        const message =
          explained.gradcam_error ||
          'The prediction succeeded, but Grad-CAM could not be generated.';
        setGradcamError(message);
        toast.error(message);
      }
    } catch (error) {
      if (error?.name === 'AbortError') return;
      const message =
        error instanceof PredictionApiError
          ? error.message
          : 'Grad-CAM could not be generated. The prediction result is unchanged.';
      setGradcamError(message);
      toast.error(message);
    }
  }

  const formError = form.formState.errors.file?.message;
  const canAnalyze = Boolean(file) && !requestPending;
  const canRetry = Boolean(file) && !requestPending && (inlineError || (prediction.isError && !result));

  return (
    <main id="main-content" className="space-y-5">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-navy">Analysis</h1>
        <p className="mt-2 max-w-2xl text-muted">
          Upload a retinal fundus image, or a camera photo of one, for a screening
          prediction (Normal, Cataract, Diabetic Retinopathy, or Glaucoma). Unsuitable
          images are rejected with: Please upload a clear image.
        </p>
        <p className="mt-2 max-w-2xl text-sm text-muted">{MODEL_SCOPE_NOTE}</p>
      </div>

      <form onSubmit={onSubmit} className="space-y-5" noValidate>
        <ImageUploader
          file={file}
          previewUrl={previewUrl}
          disabled={requestPending}
          onSelect={handleSelect}
          onRemove={handleRemove}
          onError={handleUploadError}
        />

        <div className="flex flex-wrap gap-3">
          <Button
            type="submit"
            disabled={!canAnalyze}
            aria-busy={analyzingFirstPass}
            aria-label={analyzingFirstPass ? ANALYZE_MESSAGES[loadingIndex] : 'Analyze Eye'}
          >
            {analyzingFirstPass ? (
              <>
                <LoaderCircle className="size-4 animate-spin" aria-hidden="true" />
                {ANALYZE_MESSAGES[loadingIndex]}
              </>
            ) : (
              'Analyze Eye'
            )}
          </Button>
          {canRetry ? (
            <Button type="submit" variant="secondary" disabled={!canAnalyze}>
              <RotateCcw className="size-4" aria-hidden="true" />
              Retry analysis
            </Button>
          ) : null}
          <Button type="button" variant="ghost" onClick={handleNewAnalysis}>
            New Analysis
          </Button>
          {historyId ? (
            <Button asChild variant="secondary">
              <Link to={`/history/${historyId}`}>View in History</Link>
            </Button>
          ) : null}
        </div>
      </form>

      {analyzingFirstPass ? (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <Card className="flex items-center gap-3" role="status" aria-live="polite">
            <LoaderCircle className="size-5 animate-spin text-primary" aria-hidden="true" />
            <p className="text-muted">{ANALYZE_MESSAGES[loadingIndex]}</p>
          </Card>
        </motion.div>
      ) : null}

      {inlineError || formError ? (
        <Alert variant="destructive" data-testid="analysis-error">
          <AlertDescription>
            <p>{inlineError || formError}</p>
          </AlertDescription>
        </Alert>
      ) : null}

      {result ? (
        <ResultPanel
          result={result}
          previewUrl={previewUrl}
          fileName={file?.name}
          gradcamLoading={retryingGradCam || analyzingFirstPass}
          gradcamError={gradcamError}
          onRetryGradCam={handleRetryGradCam}
        />
      ) : null}
    </main>
  );
}
