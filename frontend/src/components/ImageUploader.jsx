import { useId, useState } from 'react';
import { Upload } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { imageFileSchema } from '@/lib/validation';
import { formatFileSize, cn } from '@/lib/utils';

const ACCEPT = '.jpg,.jpeg,.png,image/jpeg,image/png';

export default function ImageUploader({
  file,
  previewUrl,
  disabled = false,
  onSelect,
  onRemove,
  onError,
}) {
  const inputId = useId();
  const [isDragging, setIsDragging] = useState(false);

  function takeFile(nextFile) {
    if (!nextFile) return;
    const parsed = imageFileSchema.safeParse(nextFile);
    if (!parsed.success) {
      onError?.(parsed.error.issues[0]?.message || 'Please choose a valid image.');
      return;
    }
    onSelect(parsed.data);
  }

  function handleInputChange(event) {
    const nextFile = event.target.files?.[0];
    event.target.value = '';
    takeFile(nextFile);
  }

  function handleDrop(event) {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);
    if (disabled) return;
    takeFile(event.dataTransfer.files?.[0]);
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle id="upload-heading">Upload fundus image</CardTitle>
        <CardDescription>JPG, JPEG, or PNG. One eye photograph per analysis.</CardDescription>
      </CardHeader>

      {!file ? (
        <label
          htmlFor={inputId}
          className={cn(
            'focus-within:ring-2 focus-within:ring-primary/40',
            'flex min-h-56 cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-[#b7cdc7] bg-[#f7fbfa] px-5 py-8 text-center transition-colors',
            isDragging && 'border-primary bg-accent',
            disabled && 'pointer-events-none opacity-60',
          )}
          onDragEnter={(event) => {
            event.preventDefault();
            if (!disabled) setIsDragging(true);
          }}
          onDragOver={(event) => {
            event.preventDefault();
            if (!disabled) setIsDragging(true);
          }}
          onDragLeave={(event) => {
            event.preventDefault();
            setIsDragging(false);
          }}
          onDrop={handleDrop}
        >
          <Upload className="size-7 text-primary" aria-hidden="true" />
          <span className="font-semibold text-navy">Drag and drop an image here</span>
          <span className="text-sm text-muted">or</span>
          <span className="inline-flex min-h-11 items-center rounded-lg border border-border bg-card px-4 text-sm font-semibold text-navy">
            Browse / Choose Image
          </span>
          <input
            id={inputId}
            className="sr-only"
            type="file"
            accept={ACCEPT}
            disabled={disabled}
            aria-labelledby="upload-heading"
            aria-describedby={`${inputId}-hint`}
            onChange={handleInputChange}
          />
          <span id={`${inputId}-hint`} className="sr-only">
            Supported formats are JPG, JPEG, and PNG.
          </span>
        </label>
      ) : (
        <div className="grid items-center gap-5 md:grid-cols-[180px_1fr]">
          <div className="overflow-hidden rounded-xl bg-navy">
            {previewUrl ? (
              <img
                src={previewUrl}
                alt="Selected fundus photograph preview"
                className="h-44 w-full object-cover"
              />
            ) : null}
          </div>
          <div>
            <p className="break-all font-semibold text-navy" title={file.name}>
              {file.name}
            </p>
            <p className="text-sm text-muted">{formatFileSize(file.size)}</p>
            <div className="mt-3 flex flex-wrap gap-2">
              <Button asChild variant="secondary" size="sm" disabled={disabled}>
                <label htmlFor={`${inputId}-change`}>
                  Change Image
                  <input
                    id={`${inputId}-change`}
                    className="sr-only"
                    type="file"
                    accept={ACCEPT}
                    disabled={disabled}
                    aria-label="Change selected fundus image"
                    onChange={handleInputChange}
                  />
                </label>
              </Button>
              <Button
                variant="ghost"
                size="sm"
                disabled={disabled}
                aria-label="Remove selected image"
                onClick={onRemove}
              >
                Remove
              </Button>
            </div>
          </div>
        </div>
      )}
    </Card>
  );
}
