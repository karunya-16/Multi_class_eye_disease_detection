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

function raiseApiError(error, fallback) {
  if (isCancelError(error)) {
    const cancelError = new Error('Request canceled');
    cancelError.name = 'AbortError';
    throw cancelError;
  }
  const status = error.response?.status ?? 0;
  const apiMessage =
    typeof error.response?.data?.error === 'string' ? error.response.data.error : null;
  let message = fallback;
  if (!error.response || error.code === 'ERR_NETWORK') {
    message = 'The screening service is unavailable. Start the FastAPI backend and try again.';
  } else if (status === 400) {
    message = 'The dashboard filters are not valid.';
  } else if (status === 500) {
    message = 'Dashboard statistics could not be loaded. Please try again.';
  }
  if (
    apiMessage &&
    apiMessage.length <= 280 &&
    !/traceback|exception:|mysql|sql syntax|pymysql|operationalerror|access denied|password|credentials|stack trace/i.test(
      apiMessage,
    )
  ) {
    message = apiMessage;
  }
  throw new PredictionApiError(message, { status });
}

function asCount(value) {
  const number = Number(value);
  return Number.isFinite(number) && number >= 0 ? number : 0;
}

function asOptionalNumber(value) {
  if (value === null || value === undefined || value === '') return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

export async function fetchDashboardAnalytics({
  predictedLabel,
  predictedClass,
  startDate,
  endDate,
  signal,
} = {}) {
  const params = {};
  if (Number.isFinite(Number(predictedClass))) params.predicted_class = Number(predictedClass);
  else if (predictedLabel) params.predicted_label = predictedLabel;
  if (startDate) params.start_date = startDate;
  if (endDate) params.end_date = endDate;

  let payload;
  try {
    const response = await axios.get(`${getApiBaseUrl()}/analytics/dashboard`, {
      params,
      signal,
      timeout: 20000,
    });
    payload = response.data;
  } catch (error) {
    raiseApiError(error, 'Dashboard statistics could not be loaded. Please try again.');
  }

  if (!payload || payload.success !== true || !Array.isArray(payload.class_distribution)) {
    throw new PredictionApiError(
      'The server returned an unexpected dashboard response. Please try again.',
      { code: 'unexpected_response' },
    );
  }

  const classDistribution = CLASS_NAMES.map((name, index) => {
    const match = payload.class_distribution.find(
      (item) => Number(item?.predicted_class) === index || item?.predicted_label === name,
    );
    return {
      predicted_class: index,
      predicted_label: name,
      count: asCount(match?.count),
    };
  });

  return {
    totalAnalyses: asCount(payload.total_analyses),
    recentAnalysisCount: asCount(payload.recent_analysis_count),
    recentWindowDays: Number(payload.recent_window_days) || 7,
    averageConfidence: asOptionalNumber(payload.average_confidence),
    averageConfidencePercentage: asOptionalNumber(payload.average_confidence_percentage),
    highestConfidence: asOptionalNumber(payload.highest_confidence),
    highestConfidencePercentage: asOptionalNumber(payload.highest_confidence_percentage),
    lowestConfidence: asOptionalNumber(payload.lowest_confidence),
    lowestConfidencePercentage: asOptionalNumber(payload.lowest_confidence_percentage),
    classDistribution,
    confidenceDistribution: Array.isArray(payload.confidence_distribution)
      ? payload.confidence_distribution.map((item) => ({
          bucket: String(item.bucket || ''),
          count: asCount(item.count),
        }))
      : [],
    activity: Array.isArray(payload.activity)
      ? payload.activity.map((item) => ({
          bucket: String(item.bucket || ''),
          count: asCount(item.count),
        }))
      : [],
    activityGranularity: payload.activity_granularity === 'hour' ? 'hour' : 'day',
    message: typeof payload.message === 'string' ? payload.message : '',
  };
}
