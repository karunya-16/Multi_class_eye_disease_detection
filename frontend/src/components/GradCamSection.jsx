import { LoaderCircle, RotateCcw, ScanSearch } from 'lucide-react';
import { motion } from 'motion/react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  GRADCAM_DISCLAIMER,
  GRADCAM_EXPLANATION,
  GRADCAM_LOADING_MESSAGE,
} from '@/lib/constants';

function isDisplayableImage(value) {
  return (
    typeof value === 'string' &&
    (value.startsWith('data:image') || value.startsWith('http://') || value.startsWith('https://'))
  );
}

function GradCamSlot({ title, src, alt }) {
  return (
    <figure className="m-0 min-w-0 rounded-xl border border-border bg-[#f7fbfa] p-3">
      {isDisplayableImage(src) ? (
        <img src={src} alt={alt} className="h-48 w-full rounded-lg object-cover sm:h-56" />
      ) : (
        <div
          className="grid h-48 place-items-center text-center text-sm text-muted sm:h-56"
          aria-label={`${title} is not available`}
        >
          Visualization not available
        </div>
      )}
      <figcaption className="mt-2 text-sm font-semibold text-navy">{title}</figcaption>
    </figure>
  );
}

export default function GradCamSection({
  result,
  loading = false,
  error = '',
  onRetry,
  historical = false,
}) {
  const cam = result?.gradcam;
  const available = result?.gradcam_available === true && cam;
  const predictedLabel = cam?.predicted_label || result?.predicted_label;
  const confidencePct = Number(
    cam?.confidence_percentage ?? result?.confidence_percentage,
  );

  return (
    <section className="space-y-4" aria-labelledby="gradcam-heading" data-testid="gradcam-section">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 id="gradcam-heading" className="flex items-center gap-2 text-base font-semibold text-navy">
            <ScanSearch className="size-4" aria-hidden="true" />
            Grad-CAM result
          </h3>
          <p className="mt-1 max-w-2xl text-sm text-muted">{GRADCAM_EXPLANATION}</p>
        </div>
        <div className="flex flex-wrap gap-2">
          {predictedLabel ? (
            <Badge data-testid="gradcam-predicted-label">Predicted class: {predictedLabel}</Badge>
          ) : null}
          {Number.isFinite(confidencePct) ? (
            <Badge data-testid="gradcam-model-confidence">
              Model Confidence {confidencePct.toFixed(2)}%
            </Badge>
          ) : null}
        </div>
      </div>

      {loading ? (
        <div
          className="flex min-h-36 items-center gap-3 rounded-xl border border-border bg-[#f7fbfa] px-4"
          role="status"
          aria-live="polite"
          data-testid="gradcam-loading"
        >
          <LoaderCircle className="size-5 animate-spin text-primary" aria-hidden="true" />
          <p className="text-muted">{GRADCAM_LOADING_MESSAGE}</p>
        </div>
      ) : null}

      {!loading && available ? (
        <motion.div
          className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.28 }}
        >
          <GradCamSlot
            title="Original Image"
            src={cam.original_image}
            alt="Original fundus photograph used for Grad-CAM"
          />
          <GradCamSlot
            title="Grad-CAM Heatmap"
            src={cam.heatmap_image}
            alt="Grad-CAM heatmap for the predicted class"
          />
          <GradCamSlot
            title="Grad-CAM Overlay"
            src={cam.overlay_image}
            alt="Grad-CAM overlay on the original fundus photograph"
          />
        </motion.div>
      ) : null}

      {!loading && available && cam?.target_layer ? (
        <p className="text-sm text-muted" data-testid="gradcam-meta">
          Target layer {cam.target_layer}
          {Array.isArray(cam.heatmap_shape) ? ` · heatmap shape (${cam.heatmap_shape.join(', ')})` : ''}
          {Number.isFinite(Number(cam.heatmap_min)) && Number.isFinite(Number(cam.heatmap_max))
            ? ` · values ${Number(cam.heatmap_min).toFixed(2)}–${Number(cam.heatmap_max).toFixed(2)}`
            : ''}
        </p>
      ) : null}

      {!loading && (error || (!available && result)) ? (
        <Alert variant={historical ? 'info' : 'destructive'} data-testid="gradcam-error">
          <AlertTitle>
            {historical ? 'Grad-CAM was not stored for this analysis.' : 'Grad-CAM could not be displayed'}
          </AlertTitle>
          <AlertDescription>
            <p>
              {historical
                ? 'This stored prediction can still be reviewed. Explainability images were not saved for this record.'
                : error ||
                  result?.gradcam_error ||
                  'The prediction is available, but Grad-CAM was not returned for this image.'}
            </p>
            {onRetry && !historical ? (
              <div className="mt-3">
                <Button type="button" variant="secondary" size="sm" onClick={onRetry}>
                  <RotateCcw className="size-4" aria-hidden="true" />
                  Retry Grad-CAM
                </Button>
              </div>
            ) : null}
          </AlertDescription>
        </Alert>
      ) : null}

      <p className="text-sm text-muted" role="note">
        {GRADCAM_DISCLAIMER} Highlighted regions are not medically confirmed disease
        regions and are not a measure of severity.
      </p>
    </section>
  );
}
