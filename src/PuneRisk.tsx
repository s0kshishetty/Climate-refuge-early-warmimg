import { useEffect, useMemo, useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

/* ── Types (match public/data/pune.json written by ml/pune_build.py) ── */

type Level = 'CRITICAL' | 'HIGH' | 'MODERATE' | 'LOW'
interface Hazard { prob: number; level: Level; lift: number }
interface Area {
  id: string; name: string; lat: number; lon: number
  elevation: number; exposure: number
  rain: Hazard; heat: Hazard; overall: Level
  recent: { rain_3d: number; rain_7d: number; rain_30d: number; tmax: number; tmin: number }
  drivers: { label: string; detail: string }[]
  history: { d: string; rain: number; tmax: number }[]
}
interface HazardMeta {
  label: string; rule: string; base_rate: number
  metrics: {
    roc_auc: number | null; pr_auc: number | null; brier: number | null
    baseline_roc_auc: number | null; baseline_pr_auc: number | null
  }
  importance: { label: string; score: number }[]
}
interface PuneData {
  meta: {
    generated: string; as_of: string; source: string; live_areas: number
    horizon_days: number; n_rows: number; level_rule: string
    train_years: number[]; val_years: number[]; test_years: number[]
    hazards: { rain: HazardMeta; heat: HazardMeta }
  }
  areas: Area[]
}

const COLORS: Record<Level, string> = {
  CRITICAL: '#B42318', HIGH: '#E8590C', MODERATE: '#C99700', LOW: '#3B7A4B',
}
const RANK: Record<Level, number> = { CRITICAL: 3, HIGH: 2, MODERATE: 1, LOW: 0 }

const pct = (p: number) => `${(p * 100).toFixed(p < 0.1 ? 1 : 0)}%`
const num = (v: number | null) => (v === null ? '–' : v.toFixed(3))

function Badge({ level }: { level: Level }) {
  return (
    <span
      className="inline-block rounded-full px-2.5 py-0.5 text-[11px] font-semibold tracking-wide text-white"
      style={{ background: COLORS[level] }}
    >
      {level}
    </span>
  )
}

function HazardTile({ title, h, note }: { title: string; h: Hazard; note: string }) {
  return (
    <div className="rounded-xl border border-black/10 bg-white p-4">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium">{title}</span>
        <Badge level={h.level} />
      </div>
      <div className="mt-3 text-3xl font-semibold" style={{ color: COLORS[h.level] }}>
        {pct(h.prob)}
      </div>
      <div className="mt-1 text-xs opacity-70">
        chance in next 3 days · {h.lift.toFixed(1)}× normal
      </div>
      <div className="mt-2 text-xs opacity-60">{note}</div>
    </div>
  )
}

export default function PuneRisk() {
  const [data, setData] = useState<PuneData | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [selId, setSelId] = useState<string | null>(null)

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/pune.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then((d: PuneData) => setData(d))
      .catch((e) => setErr(String(e)))
  }, [])

  const sorted = useMemo(
    () =>
      data
        ? [...data.areas].sort(
            (a, b) => RANK[b.overall] - RANK[a.overall] || b.rain.prob - a.rain.prob,
          )
        : [],
    [data],
  )
  const matches = useMemo(() => {
    const q = query.trim().toLowerCase()
    return q ? sorted.filter((a) => a.name.toLowerCase().includes(q)) : sorted
  }, [sorted, query])
  const selected = matches.find((a) => a.id === selId) ?? matches[0] ?? null

  if (err) {
    return (
      <div className="p-8 max-w-xl">
        <h1 className="text-xl font-semibold">Pune data not found</h1>
        <p className="mt-2 text-sm opacity-80">
          Generate it first, from the project root:
        </p>
        <pre className="mt-3 rounded-lg bg-black/5 p-3 text-sm">
{`python ml/pune_fetch.py
python ml/pune_build.py`}
        </pre>
        <p className="mt-2 text-xs opacity-60">({err})</p>
      </div>
    )
  }
  if (!data) return <div className="p-8 text-sm opacity-70">Loading Pune risk data…</div>

  const m = data.meta

  return (
    <div className="min-h-screen bg-[#F4F3EC] px-5 py-8 md:px-10" style={{ color: '#1A2418' }}>
      {/* header */}
      <div className="max-w-6xl">
        <h1 className="text-2xl font-semibold md:text-3xl">Pune · Area-wise Early Warning</h1>
        <p className="mt-1 text-sm opacity-70">
          Risk for the next {m.horizon_days} days · data as of {m.as_of} · {m.source}
        </p>
      </div>

      <div className="mt-6 grid max-w-6xl gap-6 lg:grid-cols-[320px_1fr]">
        {/* search + list */}
        <div>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search area (e.g. Kothrud, Hinjewadi)"
            className="w-full rounded-xl border border-black/15 bg-white px-4 py-2.5 text-sm outline-none focus:border-[#3B7A4B]"
          />
          <div className="mt-3 max-h-[520px] overflow-y-auto rounded-xl border border-black/10 bg-white">
            {matches.length === 0 && (
              <div className="p-4 text-sm opacity-60">No area matches “{query}”.</div>
            )}
            {matches.map((a) => (
              <button
                key={a.id}
                onClick={() => setSelId(a.id)}
                className={`flex w-full items-center justify-between border-b border-black/5 px-4 py-2.5 text-left text-sm last:border-b-0 hover:bg-black/[0.03] ${
                  selected?.id === a.id ? 'bg-black/[0.05] font-medium' : ''
                }`}
              >
                <span>{a.name}</span>
                <Badge level={a.overall} />
              </button>
            ))}
          </div>
          <p className="mt-2 text-xs opacity-60">Sorted by highest risk first.</p>
        </div>

        {/* detail */}
        {selected && (
          <div className="space-y-5">
            <div className="flex flex-wrap items-center gap-3">
              <h2 className="text-xl font-semibold">{selected.name}</h2>
              <Badge level={selected.overall} />
              <span className="text-xs opacity-60">
                {selected.lat.toFixed(3)}°N, {selected.lon.toFixed(3)}°E · {selected.elevation} m
              </span>
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <HazardTile
                title="Heavy rain / flood"
                h={selected.rain}
                note={`Scaled by area flood exposure (${selected.exposure.toFixed(1)}, estimate)`}
              />
              <HazardTile
                title="Heat stress"
                h={selected.heat}
                note={`Max temp ≥ 38 °C on any of next ${m.horizon_days} days`}
              />
            </div>

            <div className="grid gap-4 sm:grid-cols-3">
              {[
                ['Rain, last 3 days', `${selected.recent.rain_3d} mm`],
                ['Rain, last 7 days', `${selected.recent.rain_7d} mm`],
                ['Rain, last 30 days', `${selected.recent.rain_30d} mm`],
              ].map(([k, v]) => (
                <div key={k} className="rounded-xl border border-black/10 bg-white p-3">
                  <div className="text-xs opacity-60">{k}</div>
                  <div className="mt-1 text-lg font-semibold">{v}</div>
                </div>
              ))}
            </div>

            <div className="rounded-xl border border-black/10 bg-white p-4">
              <div className="mb-2 text-sm font-medium">Daily rainfall, last 30 days (mm)</div>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={selected.history}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#0001" />
                    <XAxis dataKey="d" interval={4} tick={{ fontSize: 10 }} />
                    <YAxis tick={{ fontSize: 10 }} width={32} />
                    <Tooltip />
                    <Bar dataKey="rain" fill="#2F6F8F" radius={[2, 2, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="rounded-xl border border-black/10 bg-white p-4">
              <div className="mb-2 text-sm font-medium">Why this risk?</div>
              <ul className="space-y-1.5 text-sm">
                {selected.drivers.map((d) => (
                  <li key={d.label}>
                    <span className="font-medium">{d.label}:</span>{' '}
                    <span className="opacity-80">{d.detail}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </div>

      {/* model card */}
      <div className="mt-10 max-w-6xl rounded-xl border border-black/10 bg-white p-5">
        <h3 className="text-base font-semibold">How reliable is this?</h3>
        <p className="mt-1 text-xs opacity-70">
          Trained on {m.n_rows.toLocaleString()} real area-days · train {m.train_years[0]}–
          {m.train_years[1]}, validation {m.val_years[0]}–{m.val_years[1]}, test{' '}
          {m.test_years[0]}–{m.test_years[1]}. {m.level_rule}
        </p>
        <div className="mt-4 grid gap-5 md:grid-cols-2">
          {(Object.keys(m.hazards) as (keyof PuneData['meta']['hazards'])[]).map((k) => {
            const h = m.hazards[k]
            return (
              <div key={k} className="text-sm">
                <div className="font-medium">{h.label}</div>
                <div className="text-xs opacity-60">
                  Event: {h.rule} · normally {pct(h.base_rate)} of days
                </div>
                <table className="mt-2 w-full text-xs">
                  <thead>
                    <tr className="text-left opacity-60">
                      <th className="py-1 font-normal">Test score</th>
                      <th className="font-normal">Model</th>
                      <th className="font-normal">Month-average baseline</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td className="py-1">ROC-AUC</td>
                      <td>{num(h.metrics.roc_auc)}</td>
                      <td>{num(h.metrics.baseline_roc_auc)}</td>
                    </tr>
                    <tr>
                      <td className="py-1">PR-AUC</td>
                      <td>{num(h.metrics.pr_auc)}</td>
                      <td>{num(h.metrics.baseline_pr_auc)}</td>
                    </tr>
                  </tbody>
                </table>
                <div className="mt-2 text-xs opacity-70">
                  Top drivers: {h.importance.map((i) => i.label).join(', ') || '–'}
                </div>
              </div>
            )
          })}
        </div>
        <p className="mt-4 text-xs opacity-60">
          Limits: weather is reanalysis data (≈10–25 km grid), so nearby areas share similar
          weather; area differences come mostly from location, elevation and the flood-exposure
          estimate. This is a research prototype, not an official warning — follow IMD / PMC
          alerts.
        </p>
      </div>
    </div>
  )
}
