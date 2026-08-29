import { z } from 'zod';

const ALLOWED_TYPES = new Set(['image/jpeg', 'image/jpg', 'image/png']);
const ALLOWED_EXT = new Set(['.jpg', '.jpeg', '.png']);

function extensionOf(name) {
  const idx = String(name || '').lastIndexOf('.');
  return idx >= 0 ? name.slice(idx).toLowerCase() : '';
}

export function isAllowedImageFile(file) {
  if (!file) return false;
  if (file.size <= 0) return false;
  const typeOk = !file.type || ALLOWED_TYPES.has(file.type.toLowerCase());
  const extOk = ALLOWED_EXT.has(extensionOf(file.name));
  return typeOk && extOk;
}

export const imageFileSchema = z
  .custom((value) => value instanceof File, {
    message: 'Please choose a JPG, JPEG, or PNG fundus image.',
  })
  .refine((file) => file.size > 0, {
    message: 'The selected file is empty. Please choose a valid image.',
  })
  .refine((file) => isAllowedImageFile(file), {
    message: 'Unsupported file type. Please upload a JPG, JPEG, or PNG image.',
  });

export const analysisFormSchema = z.object({
  file: imageFileSchema,
});
