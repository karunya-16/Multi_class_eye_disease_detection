/** Allowed camera / gallery formats for Analysis uploads. */

export const ALLOWED_IMAGE_EXTENSIONS = [
  '.jpg',
  '.jpeg',
  '.png',
  '.webp',
  '.bmp',
  '.tif',
  '.tiff',
  '.heic',
  '.heif',
];

export const ALLOWED_IMAGE_TYPES = new Set([
  'image/jpeg',
  'image/jpg',
  'image/png',
  'image/webp',
  'image/bmp',
  'image/tiff',
  'image/heic',
  'image/heif',
]);

export const IMAGE_ACCEPT =
  '.jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff,.heic,.heif,image/jpeg,image/png,image/webp,image/bmp,image/tiff,image/heic';

export const CAMERA_CAPTURE = 'environment';

export const UNSUPPORTED_FILE_MESSAGE =
  'Unsupported file type. Please upload a JPG, JPEG, PNG, WEBP, BMP, TIFF, or HEIC image.';

export const EMPTY_FILE_MESSAGE = 'The selected file is empty. Please choose a valid image.';

export const CHOOSE_IMAGE_MESSAGE = 'Please choose a camera photo or a JPG, JPEG, PNG, WEBP, BMP, TIFF, or HEIC image.';
