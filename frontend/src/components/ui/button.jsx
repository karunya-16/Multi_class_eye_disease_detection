import { cva } from 'class-variance-authority';
import { Slot } from '@radix-ui/react-slot';
import { cn } from '@/lib/utils';

const buttonVariants = cva(
  'inline-flex items-center justify-center gap-2 rounded-lg text-sm font-semibold transition-colors disabled:pointer-events-none disabled:opacity-45 min-h-11 px-4',
  {
    variants: {
      variant: {
        default: 'bg-primary text-primary-foreground hover:bg-[#0b5248]',
        secondary: 'border border-border bg-card text-navy hover:bg-accent',
        ghost: 'border border-border bg-transparent text-navy hover:bg-accent',
        outline: 'border border-primary text-primary bg-card hover:bg-accent',
      },
      size: {
        default: 'min-h-11 px-4',
        sm: 'min-h-9 px-3 text-sm',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  },
);

export function Button({
  className,
  variant,
  size,
  asChild = false,
  type = 'button',
  ...props
}) {
  const Comp = asChild ? Slot : 'button';
  return (
    <Comp
      type={asChild ? undefined : type}
      className={cn(buttonVariants({ variant, size }), className)}
      {...props}
    />
  );
}
