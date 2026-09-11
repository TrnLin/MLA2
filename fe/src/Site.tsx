import { lazy, Suspense } from 'react';

const Demo = lazy(() => import('./App'));
const Landing = lazy(() => import('./Landing'));

export default function Site() {
  const isDemo = /^\/demo\/?$/.test(window.location.pathname);
  return <Suspense fallback={<p role="status" style={{ padding: 32 }}>Opening Thread…</p>}>
    {isDemo ? <Demo /> : <Landing />}
  </Suspense>;
}
