import { z } from 'zod';
import {
  ALLOWED_IMAGE_EXTENSIONS,
  ALLOWED_IMAGE_TYPES,
  CHOOSE_IMAGE_MESSAGE,
  EMPTY_FILE_MESSAGE,
  UNSUPPORTED_FILE_MESSAGE,
} from '@/lib/uploadFormats';

function extensionOf(name) {
  const idx = String(name || '').lastIndexOf('.');
  return idx >= 0 ? name.slice(idx).toLowerCase() : '';
}

export function isAllowedImageFile(file) {
  if (!file) return false;
  if (file.size <= 0) return false;
  const typeOk = !file.type || ALLOWED_IMAGE_TYPES.has(file.type.toLowerCase()) || file.type === 'image/*';
  const extOk = ALLOWED_IMAGE_EXTENSIONS.includes(extensionOf(file.name));
  return typeOk && extOk;
}

export const imageFileSchema = z
  .custom((value) => value instanceof File, {
    message: CHOOSE_IMAGE_MESSAGE,
  })
  .refine((file) => file.size > 0, {
    message: EMPTY_FILE_MESSAGE,
  })
  .refine((file) => isAllowedImageFile(file), {
    message: UNSUPPORTED_FILE_MESSAGE,
  });

export const analysisFormSchema = z.object({
  file: imageFileSchema,
});
