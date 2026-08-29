import { Link, useParams } from 'react-router-dom';
import { ArrowLeft, LoaderCircle, RotateCcw } from 'lucide-react';
import ResultPanel from '@/components/ResultPanel';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { useAnalysisDetail } from '@/hooks/useAnalysisHistory';
import { formatAnalysisTime } from '@/lib/utils';
import { PredictionApiError } from '@/services/predictionApi';

export default function HistoryDetailPage() {
  const { id } = useParams();
  const analysisId = Number(id);
  const invalidId = !Number.isFinite(analysisId) || analysisId <= 0;
  const detail = useAnalysisDetail(id);
  const errorMessage = invalidId
    ? 'That analysis was not found.'
    : detail.error instanceof PredictionApiError
      ? detail.error.message
      : detail.isError
        ? 'That analysis could not be loaded. Please try again.'
        : '';

  return (
    <main id="main-content" className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <Button asChild variant="ghost" size="sm">
          <Link to="/history">
            <ArrowLeft className="size-4" aria-hidden="true" />
            Back to History
          </Link>
        </Button>
        <Button asChild>
          <Link to="/analysis">Start New Analysis</Link>
        </Button>
      </div>

      {detail.isPending ? (
        <Card className="flex items-center gap-3" role="status">
          <LoaderCircle className="size-5 animate-spin text-primary" aria-hidden="true" />
          <p className="text-muted">Loading stored analysis...</p>
        </Card>
      ) : null}

      {errorMessage ? (
        <Alert variant="destructive">
          <AlertTitle>Stored result unavailable</AlertTitle>
          <AlertDescription>
            <p>{errorMessage}</p>
            {!invalidId ? (
              <Button type="button" variant="secondary" size="sm" className="mt-3" onClick={() => detail.refetch()}>
                <RotateCcw className="size-4" aria-hidden="true" />
                Retry
              </Button>
            ) : null}
          </AlertDescription>
        </Alert>
      ) : null}

      {detail.data ? (
        <>
          <p className="text-sm text-muted" data-testid="history-timestamp">
            Stored {formatAnalysisTime(detail.data.created_at)}
            {detail.data.filename ? ` · ${detail.data.filename}` : ''}
          </p>
          <ResultPanel
            result={detail.data}
            previewUrl={detail.data.uploaded_image || detail.data.gradcam?.original_image || ''}
            fileName={detail.data.filename}
            historical
          />
        </>
      ) : null}
    </main>
  );
}
