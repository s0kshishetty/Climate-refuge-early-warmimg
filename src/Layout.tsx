import { useState } from 'react'
import App from './App'
import PuneRisk from './PuneRisk'

type Page = 'india' | 'pune'

const ITEMS: { id: Page; label: string; sub: string }[] = [
  { id: 'india', label: 'India', sub: 'State-wise risk' },
  { id: 'pune', label: 'Pune', sub: 'Area-wise risk' },
]

export default function Layout() {
  const [page, setPage] = useState<Page>(() =>
    window.location.hash === '#pune' ? 'pune' : 'india',
  )

  const go = (p: Page) => {
    setPage(p)
    window.scrollTo(0, 0)
  }

  return (
    <div className="min-h-screen bg-[#F4F3EC]">
      {/* desktop sidebar */}
      <aside
        className="fixed inset-y-0 left-0 z-[60] hidden w-60 flex-col px-4 py-6 md:flex"
        style={{ background: '#1A2418', color: '#F4F3EC' }}
      >
        <div className="px-2 text-lg font-semibold">ClimateWatch</div>
        <div className="mb-6 px-2 text-xs opacity-60">Early Warning System</div>
        <nav className="space-y-1">
          {ITEMS.map((it) => (
            <button
              key={it.id}
              onClick={() => go(it.id)}
              className={`w-full rounded-lg px-3 py-2.5 text-left transition-colors ${
                page === it.id ? 'bg-white/15' : 'hover:bg-white/10'
              }`}
            >
              <div className="text-sm font-medium">{it.label}</div>
              <div className="text-xs opacity-60">{it.sub}</div>
            </button>
          ))}
        </nav>
      </aside>

      {/* mobile top tabs */}
      <div
        className="fixed inset-x-0 top-0 z-[60] flex h-12 items-center gap-2 px-3 md:hidden"
        style={{ background: '#1A2418', color: '#F4F3EC' }}
      >
        {ITEMS.map((it) => (
          <button
            key={it.id}
            onClick={() => go(it.id)}
            className={`rounded-full px-4 py-1.5 text-sm ${
              page === it.id ? 'bg-white/20 font-medium' : 'opacity-70'
            }`}
          >
            {it.label}
          </button>
        ))}
      </div>

      {/* content: transform keeps App's own fixed navbar inside this area */}
      <main className="mt-12 md:ml-60 md:mt-0" style={{ transform: 'translateZ(0)' }}>
        {page === 'india' ? <App /> : <PuneRisk />}
      </main>
    </div>
  )
}
