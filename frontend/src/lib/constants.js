export const CLASS_NAMES = ['Normal', 'Cataract', 'Diabetic Retinopathy', 'Glaucoma'];

export const MEDICAL_DISCLAIMER =
  'This system is an AI-based research prototype for eye disease screening. The prediction and confidence score are not a medical diagnosis and should not be interpreted as disease severity or percentage of eye damage. Please consult a qualified eye-care professional for clinical evaluation.';

export const MODEL_SCOPE_NOTE =
  'The current EfficientNetV2-B0 model is validated on retinal fundus images (84.94% test accuracy). Camera and gallery uploads are an input-validation and preprocessing feature. They do not mean the model was trained on arbitrary smartphone photographs of the external eye.';

export const CLEAR_IMAGE_MESSAGE = 'Please upload a clear image';

export const LOADING_MESSAGES = [
  'Uploading image...',
  'Analyzing retinal image...',
  'Generating prediction...',
];

export const GRADCAM_LOADING_MESSAGE = 'Generating Grad-CAM...';

export const GRADCAM_EXPLANATION =
  "Grad-CAM highlights image regions that contributed to the model's prediction.";

export const GRADCAM_DISCLAIMER =
  'Grad-CAM is an explainability visualization and should not be interpreted as a clinical diagnosis or exact disease boundary.';
