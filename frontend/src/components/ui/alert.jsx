import { cn } from '@/lib/utils';

export function Alert({ className, variant = 'default', ...props }) {
  return (
    <div
      role="alert"
      className={cn(
        'rounded-xl border px-4 py-3 text-sm',
        variant === 'destructive' && 'border-[#ead4d4] bg-[#f8ecec] text-destructive',
        variant === 'info' && 'border-[#cfe3de] bg-[#eef6f4] text-navy',
        variant === 'default' && 'border-border bg-accent text-navy',
        className,
      )}
      {...props}
    />
  );
}

export function AlertTitle({ className, ...props }) {
  return <p className={cn('font-semibold', className)} {...props} />;
}

export function AlertDescription({ className, ...props }) {
  return <div className={cn('mt-1 space-y-1 text-sm', className)} {...props} />;
}
