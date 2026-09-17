import { motion } from 'motion/react';
import { Activity, BadgeCheck } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import ProbabilityChart from '@/components/ProbabilityChart';
import GradCamSection from '@/components/GradCamSection';
import { CLASS_NAMES, MEDICAL_DISCLAIMER } from '@/lib/constants';
import { probabilityToPercent } from '@/lib/utils';

export default function ResultPanel({
  result,
  previewUrl,
  fileName,
  gradcamLoading = false,
  gradcamError = '',
  onRetryGradCam,
  historical = false,
}) {
  const probabilities = result.probabilities || result.class_probabilities || {};
  const disease = result.disease || result.predicted_label;
  const confidencePct = Number(
    Number.isFinite(Number(result.confidence_percentage))
      ? result.confidence_percentage
      : Number(result.confidence) * 100,
  );
  const probabilitySum = CLASS_NAMES.reduce(
    (sum, name) => sum + Number(probabilities[name] || 0),
    0,
  );

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.28 }}
    >
      <Card className="space-y-6" aria-live="polite" data-testid="prediction-result">
        <CardHeader>
          <CardTitle id="result-heading">Screening result</CardTitle>
          <CardDescription>
            Values below are the JSON fields returned by EfficientNetV2-B0 through FastAPI.
            This is not a diagnosis.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid gap-5 md:grid-cols-[220px_1fr]">
            <figure className="m-0">
              {previewUrl ? (
                <img
                  src={previewUrl}
                  alt="Analyzed fundus photograph"
                  className="h-52 w-full rounded-xl object-cover"
                />
              ) : null}
              <figcaption className="mt-2 break-all text-sm text-muted">
                {fileName || result.filename || 'Uploaded image'}
              </figcaption>
            </figure>
            <div className="grid gap-3">
              <div className="rounded-xl border border-border bg-[#f7fbfa] p-4">
                <p className="text-sm text-muted">Predicted disease</p>
                <p
                  className="text-2xl font-bold text-navy"
                  data-testid="predicted-label"
                >
                  {disease}
                </p>
              </div>
              <div className="rounded-xl border border-border bg-[#f7fbfa] p-4">
                <p className="flex items-center gap-2 text-sm text-muted">
                  <Activity className="size-4" aria-hidden="true" />
                  Model Confidence
                </p>
                <p
                  className="text-2xl font-bold text-navy"
                  data-testid="model-confidence"
                >
                  {confidencePct.toFixed(2)}%
                </p>
                <p className="mt-1 text-sm text-muted" data-testid="confidence-raw">
                  confidence: {Number(result.confidence).toFixed(4)}. This is not
                  disease severity.
                </p>
              </div>
            </div>
          </div>

          <div>
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <h3 className="text-base font-semibold text-navy">Disease probabilities</h3>
              <Badge className="gap-1">
                <BadgeCheck className="size-3.5" aria-hidden="true" />
                Predicted: {disease}
              </Badge>
            </div>
            <ProbabilityChart
              probabilities={probabilities}
              predictedLabel={disease}
            />
            <ul className="mt-3 grid gap-1 text-sm text-muted sm:grid-cols-2" data-testid="class-probabilities">
              {CLASS_NAMES.map((name) => (
                <li key={name} data-testid={`probability-${name}`}>
                  {name}: {Number(probabilities[name]).toFixed(4)} (
                  {probabilityToPercent(probabilities[name]).toFixed(2)}%)
                </li>
              ))}
            </ul>
            <p className="mt-2 text-sm text-muted" data-testid="probability-sum">
              Sum of class probabilities: {probabilitySum.toFixed(4)} (
              {(probabilitySum * 100).toFixed(2)}%)
            </p>
          </div>

          <div>
            <GradCamSection
              result={result}
              loading={gradcamLoading}
              error={gradcamError}
              onRetry={historical ? undefined : onRetryGradCam}
              historical={historical}
            />
          </div>

          {result.message ? (
            <p className="text-sm text-muted" data-testid="backend-message">
              {result.message}
            </p>
          ) : null}

          <aside className="border-t border-border pt-4" role="note">
            <h3 className="text-base font-semibold text-navy">Medical disclaimer</h3>
            <p className="mt-2 text-sm text-muted">{MEDICAL_DISCLAIMER}</p>
          </aside>
        </CardContent>
      </Card>
    </motion.div>
  );
}
