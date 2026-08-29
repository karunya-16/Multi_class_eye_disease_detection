import { NavLink } from 'react-router-dom';
import { Eye } from 'lucide-react';
import { cn } from '@/lib/utils';

const links = [
  { to: '/', label: 'Dashboard' },
  { to: '/analysis', label: 'Analysis' },
  { to: '/history', label: 'History' },
];

export default function AppHeader() {
  return (
    <header className="border-b border-border bg-card/90 backdrop-blur">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-md focus:bg-card focus:px-3 focus:py-2"
      >
        Skip to main content
      </a>
      <div className="mx-auto flex max-w-6xl flex-col gap-4 px-4 py-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          <div className="grid size-11 place-items-center rounded-xl border border-border bg-accent text-primary" aria-hidden="true">
            <Eye className="size-6" />
          </div>
          <div>
            <p className="text-lg font-semibold tracking-tight text-navy">Eye Disease Detection</p>
            <p className="text-sm text-muted">AI-Based Eye Disease Screening</p>
          </div>
        </div>
        <nav aria-label="Primary">
          <ul className="flex flex-wrap gap-2">
            {links.map((link) => (
              <li key={link.to}>
                <NavLink
                  to={link.to}
                  className={({ isActive }) =>
                    cn(
                      'inline-flex min-h-10 items-center rounded-lg px-3 text-sm font-semibold focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
                      isActive ? 'bg-primary text-primary-foreground' : 'text-navy hover:bg-accent',
                    )
                  }
                  end={link.to === '/'}
                >
                  {link.label}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>
      </div>
    </header>
  );
}
