import { Outlet } from 'react-router-dom';
import AppHeader from '@/components/AppHeader';

export default function SiteLayout() {
  return (
    <div className="min-h-screen overflow-x-hidden">
      <AppHeader />
      <div className="mx-auto max-w-6xl px-4 py-8 sm:py-10">
        <Outlet />
      </div>
    </div>
  );
}
