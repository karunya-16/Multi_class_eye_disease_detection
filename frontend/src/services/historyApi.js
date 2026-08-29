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

function isValidListItem(item) {
  return (
    item &&
    Number.isFinite(Number(item.id)) &&
    typeof item.predicted_label === 'string' &&
    CLASS_NAMES.includes(item.predicted_label) &&
    Number.isFinite(Number(item.confidence_percentage))
  );
}

export function dataUrlToFile(dataUrl, filename) {
  if (typeof dataUrl !== 'string' || !dataUrl.startsWith('data:') || !dataUrl.includes(',')) {
    return null;
  }
  const [header, data] = dataUrl.split(',');
  if (!data) return null;
  const mime = /data:([^;]+)/.exec(header)?.[1] || 'image/png';
  const binary = atob(data);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new File([bytes], filename, { type: mime });
}

export async function saveAnalysisResult({
  file,
  requestId,
  prediction,
  gradcam = null,
  signal,
} = {}) {
  if (!(file instanceof File) || file.size <= 0) {
    throw new PredictionApiError('The analysis result could not be saved. Please try again.', {
      code: 'invalid_file',
    });
  }
  if (!requestId || !prediction) {
    throw new PredictionApiError('The analysis result could not be saved. Please try again.', {
      code: 'invalid_payload',
    });
  }

  const probabilities = prediction.class_probabilities || {};
  const formData = new FormData();
  formData.append('file', file);
  formData.append('request_id', requestId);
  formData.append('predicted_class', String(prediction.predicted_class));
  formData.append('predicted_label', String(prediction.predicted_label));
  formData.append('confidence', String(prediction.confidence));
  formData.append('confidence_percentage', String(prediction.confidence_percentage));
  formData.append('class_0_probability', String(probabilities['Class 0']));
  formData.append('class_1_probability', String(probabilities['Class 1']));
  formData.append('class_2_probability', String(probabilities['Class 2']));
  formData.append('class_3_probability', String(probabilities['Class 3']));
  formData.append('class_4_probability', String(probabilities['Class 4']));

  const original = dataUrlToFile(gradcam?.original_image, 'original.png');
  const heatmap = dataUrlToFile(gradcam?.heatmap_image, 'heatmap.png');
  const overlay = dataUrlToFile(gradcam?.overlay_image, 'overlay.png');
  if (original) formData.append('original', original);
  if (heatmap) formData.append('heatmap', heatmap);
  if (overlay) formData.append('overlay', overlay);
  if (Array.isArray(gradcam?.heatmap_shape)) {
    formData.append('heatmap_shape', JSON.stringify(gradcam.heatmap_shape));
  }
  if (typeof gradcam?.target_layer === 'string' && gradcam.target_layer.trim()) {
    formData.append('target_layer', gradcam.target_layer.trim());
  }

  let payload;
  try {
    const response = await axios.post(`${getApiBaseUrl()}/history`, formData, {
      signal,
      timeout: 60000,
    });
    payload = response.data;
  } catch (error) {
    mapFailure(error, 'The analysis result could not be saved to history. Please try again.');
  }

  if (!payload || payload.success !== true || !Number.isFinite(Number(payload.id))) {
    throw new PredictionApiError('The analysis result could not be saved to history. Please try again.', {
      code: 'unexpected_response',
    });
  }
  return payload;
}

export async function fetchAnalysisHistory({
  predictedLabel,
  predictedClass,
  query,
  startDate,
  endDate,
  page = 1,
  pageSize = 8,
  signal,
} = {}) {
  const params = { page, page_size: pageSize };
  if (Number.isFinite(Number(predictedClass))) params.predicted_class = Number(predictedClass);
  else if (predictedLabel) params.predicted_label = predictedLabel;
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
  return {
    items: payload.items.filter(isValidListItem),
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
    payload = response.data;
  } catch (error) {
    mapFailure(error, 'That analysis could not be loaded. Please try again.');
  }
  if (
    !payload ||
    payload.success !== true ||
    typeof payload.predicted_label !== 'string' ||
    !payload.class_probabilities ||
    typeof payload.class_probabilities !== 'object'
  ) {
    throw new PredictionApiError(
      'This history record is incomplete and cannot be displayed.',
      { code: 'unexpected_response' },
    );
  }
  return payload;
}
