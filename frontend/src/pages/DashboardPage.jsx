import { Link } from 'react-router-dom';
import { useEffect, useMemo, useState } from 'react';
import { motion } from 'motion/react';
import {
  Activity,
  ChevronLeft,
  ChevronRight,
  Clock3,
  FolderOpen,
  LayoutDashboard,
  RotateCcw,
  TrendingDown,
  TrendingUp,
} from 'lucide-react';
import AnalysisActivityChart from '@/components/AnalysisActivityChart';
import ClassDistributionChart from '@/components/ClassDistributionChart';
import ConfidenceDistributionChart from '@/components/ConfidenceDistributionChart';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Card, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Skeleton } from '@/components/ui/skeleton';
import { useAnalysisHistory } from '@/hooks/useAnalysisHistory';
import { useDashboardAnalytics } from '@/hooks/useDashboard';
import { CLASS_NAMES, MEDICAL_DISCLAIMER } from '@/lib/constants';
import { formatAnalysisTime } from '@/lib/utils';
import { PredictionApiError } from '@/services/predictionApi';

const RECENT_PAGE_SIZE = 5;

function formatConfidencePct(value) {
  if (!Number.isFinite(Number(value))) return '—';
  return `${Number(value).toFixed(2)}%`;
}

function StatCard({ title, value, hint, icon: Icon }) {
  return (
    <Card className="min-w-0">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-sm text-muted">{title}</p>
          <p className="mt-2 text-2xl font-bold tracking-tight text-navy">{value}</p>
          {hint ? <p className="mt-1 text-xs text-muted">{hint}</p> : null}
        </div>
        <div className="grid size-10 place-items-center rounded-xl bg-accent text-primary">
          <Icon className="size-5" aria-hidden="true" />
        </div>
      </div>
    </Card>
  );
}

function DashboardSkeleton() {
  return (
    <div className="space-y-5" role="status" aria-live="polite">
      <p className="sr-only">Loading dashboard statistics...</p>
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
        {Array.from({ length: 5 }).map((_, index) => (
          <Card key={index}>
            <Skeleton className="h-4 w-28" />
            <Skeleton className="mt-3 h-8 w-20" />
            <Skeleton className="mt-2 h-3 w-36" />
          </Card>
        ))}
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <Skeleton className="h-5 w-40" />
          <Skeleton className="mt-4 h-64 w-full" />
        </Card>
        <Card>
          <Skeleton className="h-5 w-48" />
          <Skeleton className="mt-4 h-64 w-full" />
        </Card>
      </div>
      <Card>
        <Skeleton className="h-5 w-44" />
        <Skeleton className="mt-4 h-64 w-full" />
      </Card>
    </div>
  );
}

export default function DashboardPage() {
  const [predictedLabel, setPredictedLabel] = useState('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [page, setPage] = useState(1);

  useEffect(() => {
    setPage(1);
  }, [predictedLabel, startDate, endDate]);

  const invalidRange = Boolean(startDate && endDate && startDate > endDate);

  const analytics = useDashboardAnalytics({
    predictedLabel,
    startDate,
    endDate,
    enabled: !invalidRange,
  });
  const recent = useAnalysisHistory({
    predictedLabel,
    startDate,
    endDate,
    page,
    pageSize: RECENT_PAGE_SIZE,
    enabled: !invalidRange,
  });

  const stats = analytics.data;
  const recentItems = useMemo(() => recent.data?.items || [], [recent.data]);
  const analyticsError =
    analytics.error instanceof PredictionApiError
      ? analytics.error.message
      : analytics.isError
        ? 'Dashboard statistics could not be loaded. Please try again.'
        : '';
  const recentError =
    recent.error instanceof PredictionApiError
      ? recent.error.message
      : recent.isError
        ? 'Recent analyses could not be loaded. Please try again.'
        : '';
  const hasFilters = Boolean(predictedLabel || startDate || endDate);
  const isEmpty = Boolean(stats) && stats.totalAnalyses === 0;

  function retryDashboard() {
    analytics.refetch();
    recent.refetch();
  }

  function clearFilters() {
    setPredictedLabel('');
    setStartDate('');
    setEndDate('');
    setPage(1);
  }

  return (
    <main id="main-content" className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-navy">Dashboard</h1>
          <p className="mt-2 max-w-2xl text-muted">
            Live screening statistics calculated from MySQL analysis history. Model Confidence is
            the maximum softmax probability. It is not disease severity.
          </p>
        </div>
        <Button asChild>
          <Link to="/analysis">Start New Analysis</Link>
        </Button>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <label className="grid gap-1 text-sm font-medium text-navy">
          Predicted class
          <select
            className="min-h-11 rounded-lg border border-border bg-card px-3 text-sm text-navy focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
            value={predictedLabel}
            onChange={(event) => setPredictedLabel(event.target.value)}
            aria-label="Filter dashboard by predicted class"
          >
            <option value="">All classes</option>
            {CLASS_NAMES.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </label>
        <label className="grid gap-1 text-sm font-medium text-navy">
          From
          <Input
            type="date"
            value={startDate}
            onChange={(event) => setStartDate(event.target.value)}
            aria-label="Filter dashboard from date"
          />
        </label>
        <label className="grid gap-1 text-sm font-medium text-navy">
          To
          <Input
            type="date"
            value={endDate}
            onChange={(event) => setEndDate(event.target.value)}
            aria-label="Filter dashboard to date"
          />
        </label>
        <div className="flex items-end">
          <Button type="button" variant="secondary" onClick={clearFilters} disabled={!hasFilters}>
            Clear filters
          </Button>
        </div>
      </div>

      {invalidRange ? (
        <Alert variant="destructive">
          <AlertTitle>The date range is not valid.</AlertTitle>
          <AlertDescription>
            <p>Choose a start date that is on or before the end date.</p>
          </AlertDescription>
        </Alert>
      ) : null}

      {analyticsError ? (
        <Alert variant="destructive">
          <AlertTitle>Dashboard could not be loaded</AlertTitle>
          <AlertDescription>
            <p>{analyticsError}</p>
            <Button type="button" variant="secondary" size="sm" className="mt-3" onClick={retryDashboard}>
              <RotateCcw className="size-4" aria-hidden="true" />
              Retry
            </Button>
          </AlertDescription>
        </Alert>
      ) : null}

      {!invalidRange && analytics.isPending ? <DashboardSkeleton /> : null}

      {!invalidRange && !analytics.isPending && !analyticsError && isEmpty ? (
        <Card className="space-y-4 py-8 text-center">
          <FolderOpen className="mx-auto size-10 text-primary" aria-hidden="true" />
          <CardHeader className="items-center">
            <CardTitle>
              {hasFilters ? 'No analyses match these filters.' : 'No analysis data available yet.'}
            </CardTitle>
            <CardDescription>
              Completed eye analyses saved to MySQL will appear here as live statistics and charts.
            </CardDescription>
          </CardHeader>
          {hasFilters ? (
            <Button type="button" onClick={clearFilters}>
              Clear filters
            </Button>
          ) : (
            <Button asChild>
              <Link to="/analysis">Start New Analysis</Link>
            </Button>
          )}
        </Card>
      ) : null}

      {!invalidRange && !analytics.isPending && !analyticsError && stats && !isEmpty ? (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-5">
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5" data-testid="dashboard-stats">
            <StatCard
              title="Total analyses"
              value={stats.totalAnalyses}
              hint="Successful predictions stored in MySQL"
              icon={LayoutDashboard}
            />
            <StatCard
              title="Average Model Confidence"
              value={formatConfidencePct(stats.averageConfidencePercentage)}
              hint="Not disease severity"
              icon={Activity}
            />
            <StatCard
              title="Highest Model Confidence"
              value={formatConfidencePct(stats.highestConfidencePercentage)}
              hint="Maximum softmax probability stored"
              icon={TrendingUp}
            />
            <StatCard
              title="Lowest Model Confidence"
              value={formatConfidencePct(stats.lowestConfidencePercentage)}
              hint="Minimum softmax probability stored"
              icon={TrendingDown}
            />
            <StatCard
              title={`Last ${stats.recentWindowDays} days`}
              value={stats.recentAnalysisCount}
              hint="Recent analyses in the current filters"
              icon={Clock3}
            />
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Class distribution</CardTitle>
                <CardDescription>Number of stored predictions for Class 0 through Class 4.</CardDescription>
              </CardHeader>
              <ClassDistributionChart data={stats.classDistribution} />
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Model Confidence distribution</CardTitle>
                <CardDescription>
                  Count of analyses by Model Confidence range. This is not percentage of eye damage.
                </CardDescription>
              </CardHeader>
              <ConfidenceDistributionChart data={stats.confidenceDistribution} />
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Analysis activity</CardTitle>
              <CardDescription>
                Analyses over time from MySQL timestamps
                {stats.activityGranularity === 'hour' ? ', grouped by hour.' : ', grouped by day.'}
              </CardDescription>
            </CardHeader>
            {stats.activity.length ? (
              <AnalysisActivityChart data={stats.activity} granularity={stats.activityGranularity} />
            ) : (
              <p className="text-sm text-muted">No timestamped analyses are available for this range.</p>
            )}
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Recent analyses</CardTitle>
              <CardDescription>Newest stored results first. Open a record to review the full screening result.</CardDescription>
            </CardHeader>
            {recent.isPending ? (
              <div className="space-y-3" role="status">
                <Skeleton className="h-10 w-full" />
                <Skeleton className="h-10 w-full" />
                <Skeleton className="h-10 w-full" />
              </div>
            ) : null}
            {recentError ? (
              <Alert variant="destructive">
                <AlertTitle>Recent analyses could not be loaded</AlertTitle>
                <AlertDescription>
                  <p>{recentError}</p>
                  <Button type="button" variant="secondary" size="sm" className="mt-3" onClick={() => recent.refetch()}>
                    <RotateCcw className="size-4" aria-hidden="true" />
                    Retry
                  </Button>
                </AlertDescription>
              </Alert>
            ) : null}
            {!recent.isPending && !recentError && recentItems.length === 0 ? (
              <p className="text-sm text-muted">No recent analyses are available for these filters.</p>
            ) : null}
            {!recent.isPending && recentItems.length > 0 ? (
              <>
                <div className="hidden overflow-x-auto md:block">
                  <table className="min-w-full text-left text-sm">
                    <thead className="border-b border-border text-navy">
                      <tr>
                        <th className="px-2 py-3 font-semibold">Date / time</th>
                        <th className="px-2 py-3 font-semibold">Predicted class</th>
                        <th className="px-2 py-3 font-semibold">Model Confidence</th>
                        <th className="px-2 py-3 font-semibold"><span className="sr-only">Actions</span></th>
                      </tr>
                    </thead>
                    <tbody>
                      {recentItems.map((item) => (
                        <tr key={item.id} className="border-b border-border last:border-b-0">
                          <td className="px-2 py-3 text-muted">{formatAnalysisTime(item.created_at)}</td>
                          <td className="px-2 py-3 font-semibold text-navy">{item.predicted_label}</td>
                          <td className="px-2 py-3 text-navy">
                            {Number(item.confidence_percentage).toFixed(2)}%
                          </td>
                          <td className="px-2 py-3 text-right">
                            <Button asChild variant="secondary" size="sm">
                              <Link to={`/history/${item.id}`}>View Details</Link>
                            </Button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <ul className="grid gap-3 md:hidden">
                  {recentItems.map((item) => (
                    <li key={item.id} className="rounded-xl border border-border p-4">
                      <p className="text-sm text-muted">{formatAnalysisTime(item.created_at)}</p>
                      <p className="mt-1 text-lg font-semibold text-navy">{item.predicted_label}</p>
                      <p className="text-sm text-muted">
                        Model Confidence {Number(item.confidence_percentage).toFixed(2)}%
                      </p>
                      <Button asChild size="sm" className="mt-3">
                        <Link to={`/history/${item.id}`}>View Details</Link>
                      </Button>
                    </li>
                  ))}
                </ul>
                <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
                  <p className="text-sm text-muted">
                    Page {page} of {recent.data?.totalPages || 1}
                    {recent.data?.total
                      ? ` · ${recent.data.total} matching ${recent.data.total === 1 ? 'analysis' : 'analyses'}`
                      : ''}
                  </p>
                  <div className="flex gap-2">
                    <Button
                      type="button"
                      variant="secondary"
                      size="sm"
                      disabled={page <= 1 || recent.isFetching}
                      onClick={() => setPage((current) => Math.max(1, current - 1))}
                    >
                      <ChevronLeft className="size-4" aria-hidden="true" />
                      Previous
                    </Button>
                    <Button
                      type="button"
                      variant="secondary"
                      size="sm"
                      disabled={page >= (recent.data?.totalPages || 1) || recent.isFetching}
                      onClick={() => setPage((current) => current + 1)}
                    >
                      Next
                      <ChevronRight className="size-4" aria-hidden="true" />
                    </Button>
                  </div>
                </div>
              </>
            ) : null}
          </Card>
        </motion.div>
      ) : null}

      <aside className="border-t border-border pt-4" role="note">
        <h2 className="text-base font-semibold text-navy">Medical disclaimer</h2>
        <p className="mt-2 text-sm text-muted">{MEDICAL_DISCLAIMER}</p>
      </aside>
    </main>
  );
}
