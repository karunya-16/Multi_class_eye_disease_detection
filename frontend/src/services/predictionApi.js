import axios from 'axios';
import { CLASS_NAMES } from '@/lib/constants';

const DEFAULT_API_BASE_URL = 'http://127.0.0.1:8000';
const PREDICT_TIMEOUT_MS = 120000;

export function getApiBaseUrl() {
  const raw = import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE_URL;
  return String(raw).replace(/\/$/, '');
}

export class PredictionApiError extends Error {
  constructor(message, { status = 0, code = 'error' } = {}) {
    super(message);
    this.name = 'PredictionApiError';
    this.status = status;
    this.code = code;
  }
}

function messageForFailure(error) {
  const status = error.response?.status;
  const code = error.code;

  if (code === 'ECONNABORTED' || code === 'ETIMEDOUT' || /timeout/i.test(String(error.message || ''))) {
    return {
      message: 'The prediction request timed out. Please try again.',
      status: status ?? 0,
      code: 'timeout',
    };
  }

  if (!error.response || code === 'ERR_NETWORK') {
    return {
      message: 'The screening service is unavailable. Start the FastAPI backend and try again.',
      status: 0,
      code: 'unavailable',
    };
  }

  if (status === 400) {
    return {
      message:
        'The image could not be analyzed. Please upload a valid JPG, JPEG, or PNG fundus photograph.',
      status,
      code: 'bad_request',
    };
  }

  if (status === 500) {
    return {
      message: 'The screening service could not complete this analysis. Please try again.',
      status,
      code: 'server_error',
    };
  }

  return {
    message: 'The request could not be completed. Please try again.',
    status: status ?? 0,
    code: 'error',
  };
}

function safeUserMessage(raw, fallback) {
  if (typeof raw !== 'string') return fallback;
  const trimmed = raw.trim();
  if (!trimmed) return fallback;
  if (trimmed.length > 280) return fallback;
  if (/traceback|exception:|stack trace|file ".+:\d+|line \d+/i.test(trimmed)) {
    return fallback;
  }
  return trimmed;
}

function isCancelError(error) {
  return (
    axios.isCancel?.(error) ||
    error?.code === 'ERR_CANCELED' ||
    error?.name === 'CanceledError' ||
    error?.name === 'AbortError'
  );
}

function isValidPredictionPayload(payload) {
  if (!payload || typeof payload !== 'object' || payload.success !== true) return false;
  if (typeof payload.predicted_label !== 'string') return false;
  if (!Number.isFinite(Number(payload.predicted_class))) return false;
  if (!Number.isFinite(Number(payload.confidence))) return false;
  if (!Number.isFinite(Number(payload.confidence_percentage))) return false;
  if (!payload.class_probabilities || typeof payload.class_probabilities !== 'object') {
    return false;
  }
  return CLASS_NAMES.every((name) =>
    Number.isFinite(Number(payload.class_probabilities[name])),
  );
}

export async function predictEyeDisease(file, { signal } = {}) {
  if (!(file instanceof File) || file.size <= 0) {
    throw new PredictionApiError('Please choose a valid JPG, JPEG, or PNG fundus image.', {
      status: 0,
      code: 'invalid_file',
    });
  }

  const formData = new FormData();
  formData.append('file', file);

  let payload;
  try {
    const response = await axios.post(`${getApiBaseUrl()}/predict`, formData, {
      signal,
      timeout: PREDICT_TIMEOUT_MS,
    });
    payload = response.data;
  } catch (error) {
    if (isCancelError(error)) {
      const cancelError = new Error('Request canceled');
      cancelError.name = 'AbortError';
      throw cancelError;
    }

    const mapped = messageForFailure(error);
    const apiMessage = safeUserMessage(error.response?.data?.error, null);
    throw new PredictionApiError(apiMessage || mapped.message, {
      status: mapped.status,
      code: mapped.code,
    });
  }

  if (!isValidPredictionPayload(payload)) {
    throw new PredictionApiError(
      'The server returned an unexpected response. Please try again.',
      { status: 0, code: 'unexpected_response' },
    );
  }

  return payload;
}

function isDisplayableImage(value) {
  return typeof value === 'string' && value.startsWith('data:image');
}

function isValidExplainPayload(payload) {
  if (!isValidPredictionPayload(payload)) return false;
  if (payload.gradcam_available === true) {
    const cam = payload.gradcam;
    if (!cam || typeof cam !== 'object') return false;
    return (
      isDisplayableImage(cam.original_image) &&
      isDisplayableImage(cam.heatmap_image) &&
      isDisplayableImage(cam.overlay_image)
    );
  }
  return true;
}

export async function explainEyeDisease(file, { signal } = {}) {
  if (!(file instanceof File) || file.size <= 0) {
    throw new PredictionApiError('Please choose a valid JPG, JPEG, or PNG fundus image.', {
      status: 0,
      code: 'invalid_file',
    });
  }

  const formData = new FormData();
  formData.append('file', file);

  let payload;
  try {
    const response = await axios.post(`${getApiBaseUrl()}/explain`, formData, {
      signal,
      timeout: PREDICT_TIMEOUT_MS,
    });
    payload = response.data;
  } catch (error) {
    if (isCancelError(error)) {
      const cancelError = new Error('Request canceled');
      cancelError.name = 'AbortError';
      throw cancelError;
    }

    const mapped = messageForFailure(error);
    const apiMessage = safeUserMessage(error.response?.data?.error, null);
    throw new PredictionApiError(apiMessage || mapped.message, {
      status: mapped.status,
      code: mapped.code === 'server_error' ? 'gradcam_request_failed' : mapped.code,
    });
  }

  if (!isValidExplainPayload(payload)) {
    throw new PredictionApiError(
      'The server returned an unexpected explainability response. Please try again.',
      { status: 0, code: 'unexpected_response' },
    );
  }

  return payload;
}
