import { Link } from 'react-router-dom';
import { useEffect, useMemo, useState } from 'react';
import { ChevronLeft, ChevronRight, Clock3, FolderOpen, LoaderCircle, RotateCcw, Search } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { useAnalysisHistory } from '@/hooks/useAnalysisHistory';
import { CLASS_NAMES } from '@/lib/constants';
import { formatAnalysisTime } from '@/lib/utils';
import { PredictionApiError } from '@/services/predictionApi';

const PAGE_SIZE = 8;

export default function HistoryPage() {
  const [predictedLabel, setPredictedLabel] = useState('');
  const [search, setSearch] = useState('');
  const [page, setPage] = useState(1);
  const history = useAnalysisHistory({
    predictedLabel,
    query: search.trim(),
    page,
    pageSize: PAGE_SIZE,
  });

  useEffect(() => {
    setPage(1);
  }, [predictedLabel, search]);

  const listing = history.data;
  const items = useMemo(() => listing?.items || [], [listing]);
  const total = listing?.total || 0;
  const totalPages = listing?.totalPages || 1;
  const errorMessage =
    history.error instanceof PredictionApiError
      ? history.error.message
      : history.isError
        ? 'Analysis history could not be loaded. Please try again.'
        : '';
  const hasFilters = Boolean(predictedLabel || search.trim());
  const emptyMessage = hasFilters
    ? 'No analyses match this predicted disease or search.'
    : 'No analysis history available yet.';

  return (
    <main id="main-content" className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-navy">History</h1>
          <p className="mt-2 max-w-2xl text-muted">
            Previous screening results from this application, newest first. Model Confidence is not
            disease severity.
          </p>
        </div>
        <Button asChild>
          <Link to="/analysis">Start New Analysis</Link>
        </Button>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-[220px_1fr_auto]">
        <label className="grid gap-1 text-sm font-medium text-navy">
          Predicted disease
          <select
            className="min-h-11 rounded-lg border border-border bg-card px-3 text-sm text-navy focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
            value={predictedLabel}
            onChange={(event) => setPredictedLabel(event.target.value)}
            aria-label="Filter history by predicted disease"
          >
            <option value="">All diseases</option>
            {CLASS_NAMES.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </label>
        <label className="grid gap-1 text-sm font-medium text-navy">
          Search filename
          <span className="relative">
            <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted" aria-hidden="true" />
            <Input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search by image filename"
              className="pl-9"
              aria-label="Search analysis history by filename"
            />
          </span>
        </label>
      </div>

      {history.isPending ? (
        <Card className="flex items-center gap-3" role="status">
          <LoaderCircle className="size-5 animate-spin text-primary" aria-hidden="true" />
          <p className="text-muted">Loading analysis history...</p>
        </Card>
      ) : null}

      {errorMessage ? (
        <Alert variant="destructive">
          <AlertTitle>History could not be loaded</AlertTitle>
          <AlertDescription>
            <p>{errorMessage}</p>
            <Button type="button" variant="secondary" size="sm" className="mt-3" onClick={() => history.refetch()}>
              <RotateCcw className="size-4" aria-hidden="true" />
              Retry
            </Button>
          </AlertDescription>
        </Alert>
      ) : null}

      {!history.isPending && !errorMessage && items.length === 0 ? (
        <Card className="space-y-4 py-8 text-center">
          <FolderOpen className="mx-auto size-10 text-primary" aria-hidden="true" />
          <CardHeader className="items-center">
            <CardTitle>{emptyMessage}</CardTitle>
            <CardDescription>
              Completed eye analyses from this application will appear here, newest first.
            </CardDescription>
          </CardHeader>
          <Button asChild>
            <Link to="/analysis">Start New Analysis</Link>
          </Button>
        </Card>
      ) : null}

      {!history.isPending && items.length > 0 ? (
        <>
          <div className="hidden overflow-x-auto rounded-2xl border border-border bg-card md:block" data-testid="history-table">
            <table className="min-w-full text-left text-sm">
              <thead className="border-b border-border bg-[#f7fbfa] text-navy">
                <tr>
                  <th className="px-4 py-3 font-semibold">Date / time</th>
                  <th className="px-4 py-3 font-semibold">Predicted disease</th>
                  <th className="px-4 py-3 font-semibold">Model Confidence</th>
                  <th className="px-4 py-3 font-semibold">Grad-CAM</th>
                  <th className="px-4 py-3 font-semibold"><span className="sr-only">Actions</span></th>
                </tr>
              </thead>
              <tbody>
                {items.map((item) => (
                  <tr key={item.id} className="border-b border-border last:border-b-0">
                    <td className="px-4 py-3 text-muted">{formatAnalysisTime(item.created_at)}</td>
                    <td className="px-4 py-3 font-semibold text-navy">{item.disease}</td>
                    <td className="px-4 py-3 text-navy">{Number(item.confidence_percentage).toFixed(2)}%</td>
                    <td className="px-4 py-3 text-muted">
                      {item.gradcam_available ? 'Available' : 'Not stored'}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <Button asChild variant="secondary" size="sm">
                        <Link to={`/history/${item.id}`}>View Details</Link>
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <ul className="grid gap-3 md:hidden" data-testid="history-cards">
            {items.map((item) => (
              <li key={item.id}>
                <Card>
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="flex items-center gap-2 text-sm text-muted">
                        <Clock3 className="size-4" aria-hidden="true" />
                        {formatAnalysisTime(item.created_at)}
                      </p>
                      <p className="mt-2 text-lg font-semibold text-navy">{item.disease}</p>
                      <p className="text-sm text-muted">
                        Model Confidence {Number(item.confidence_percentage).toFixed(2)}%
                      </p>
                    </div>
                    <Badge>{item.gradcam_available ? 'Grad-CAM' : 'No Grad-CAM'}</Badge>
                  </div>
                  <div className="mt-4">
                    <Button asChild size="sm">
                      <Link to={`/history/${item.id}`}>View Details</Link>
                    </Button>
                  </div>
                </Card>
              </li>
            ))}
          </ul>

          <div className="flex flex-wrap items-center justify-between gap-3" data-testid="history-pagination">
            <p className="text-sm text-muted">
              Page {page} of {totalPages}
              {total ? ` · ${total} stored ${total === 1 ? 'analysis' : 'analyses'}` : ''}
            </p>
            <div className="flex gap-2">
              <Button
                type="button"
                variant="secondary"
                size="sm"
                disabled={page <= 1 || history.isFetching}
                onClick={() => setPage((current) => Math.max(1, current - 1))}
              >
                <ChevronLeft className="size-4" aria-hidden="true" />
                Previous
              </Button>
              <Button
                type="button"
                variant="secondary"
                size="sm"
                disabled={page >= totalPages || history.isFetching}
                onClick={() => setPage((current) => current + 1)}
              >
                Next
                <ChevronRight className="size-4" aria-hidden="true" />
              </Button>
            </div>
          </div>
        </>
      ) : null}
    </main>
  );
}
