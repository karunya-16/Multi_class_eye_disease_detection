import axios from 'axios';
import { CLASS_NAMES } from '@/lib/constants';
import { PredictionApiError, getApiBaseUrl } from '@/services/predictionApi';

function isCancelError(error) {
  return (
    axios.isCancel?.(error) ||
    error?.code === 'ERR_CANCELED' ||
    error?.name === 'CanceledError' ||
    error?.name === 'AbortError'
  );
}

function throwCanceled() {
  const cancelError = new Error('Request canceled');
  cancelError.name = 'AbortError';
  throw cancelError;
}

function isUnsafeServerMessage(message) {
  return /traceback|exception:|mysql|sql syntax|pymysql|operationalerror|access denied|password|credentials|stack trace|file ".+:\d+/i.test(
    message,
  );
}

function mapFailure(error, fallback) {
  if (isCancelError(error)) throwCanceled();
  const status = error.response?.status ?? 0;
  const apiMessage =
    typeof error.response?.data?.error === 'string' ? error.response.data.error : null;
  let message = fallback;
  if (!error.response || error.code === 'ERR_NETWORK') {
    message = 'The screening service is unavailable. Start the FastAPI backend and try again.';
  } else if (status === 404) {
    message = 'That analysis was not found.';
  } else if (status === 400) {
    message = 'The history request was not valid.';
  } else if (status === 500) {
    message = 'The screening service could not complete this history request. Please try again.';
  }
  if (
    apiMessage &&
    apiMessage.length <= 280 &&
    !isUnsafeServerMessage(apiMessage)
  ) {
    message = apiMessage;
  }
  throw new PredictionApiError(message, { status });
}

function round4(value) {
  return Math.round(Number(value) * 10000) / 10000;
}

export function normalizeHistoryRecord(raw) {
  if (!raw || typeof raw !== 'object') return raw;
  const probabilities = raw.probabilities || raw.class_probabilities;
  const disease = raw.disease || raw.predicted_label;
  const classId = raw.class_id ?? raw.predicted_class;
  const confidence = Number(raw.confidence);
  const confidencePercentage = Number.isFinite(Number(raw.confidence_percentage))
    ? Number(raw.confidence_percentage)
    : Math.round(confidence * 10000) / 100;
  return {
    ...raw,
    disease,
    class_id: classId,
    confidence: round4(confidence),
    probabilities,
    predicted_label: disease,
    predicted_class: classId,
    confidence_percentage: confidencePercentage,
    class_probabilities: probabilities,
  };
}

function isValidListItem(item) {
  return (
    item &&
    Number.isFinite(Number(item.id)) &&
    typeof item.disease === 'string' &&
    CLASS_NAMES.includes(item.disease) &&
    Number.isFinite(Number(item.confidence_percentage))
  );
}

export async function fetchAnalysisHistory({
  predictedLabel,
  query,
  startDate,
  endDate,
  page = 1,
  pageSize = 8,
  signal,
} = {}) {
  const params = { page, page_size: pageSize };
  if (predictedLabel) params.disease = predictedLabel;
  if (query) params.q = query;
  if (startDate) params.start_date = startDate;
  if (endDate) params.end_date = endDate;
  let payload;
  try {
    const response = await axios.get(`${getApiBaseUrl()}/history`, {
      params,
      signal,
      timeout: 20000,
    });
    payload = response.data;
  } catch (error) {
    mapFailure(error, 'Analysis history could not be loaded. Please try again.');
  }
  if (!payload || payload.success !== true || !Array.isArray(payload.items)) {
    throw new PredictionApiError(
      'The server returned an unexpected history response. Please try again.',
      { code: 'unexpected_response' },
    );
  }
  const items = payload.items.map(normalizeHistoryRecord).filter(isValidListItem);
  return {
    items,
    total: Number(payload.total) || 0,
    page: Number(payload.page) || page,
    pageSize: Number(payload.page_size) || pageSize,
    totalPages: Number(payload.total_pages) || 1,
  };
}

export async function fetchAnalysisDetail(id, { signal } = {}) {
  const analysisId = Number(id);
  if (!Number.isFinite(analysisId) || analysisId <= 0) {
    throw new PredictionApiError('That analysis was not found.', { status: 404 });
  }
  let payload;
  try {
    const response = await axios.get(`${getApiBaseUrl()}/history/${analysisId}`, {
      signal,
      timeout: 20000,
    });
    payload = normalizeHistoryRecord(response.data);
  } catch (error) {
    mapFailure(error, 'That analysis could not be loaded. Please try again.');
  }
  if (
    !payload ||
    payload.success !== true ||
    !CLASS_NAMES.includes(payload.disease) ||
    !payload.probabilities ||
    typeof payload.probabilities !== 'object'
  ) {
    throw new PredictionApiError(
      'This history record is incomplete and cannot be displayed.',
      { code: 'unexpected_response' },
    );
  }
  return payload;
}
