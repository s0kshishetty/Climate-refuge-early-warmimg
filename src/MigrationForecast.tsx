import { useEffect, useMemo, useState } from 'react'
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine, Cell,
} from 'recharts'

/* Reads public/data/mh_forecast.json written by ml/mh_predict.py */

interface Pred { share: number; low: number; high: number; climate_effect_pp: number; population: number; migrants: number }
interface Dist { id: string; name: string; share_2001: number; share_2011: number; pop_2011: number; pop_cagr_pct: number; pred: Record<string, Pred> }
interface Forecast {
  meta: {
    generated: string; n_districts: number; years: string[]; target: string; best_model: string
    beats_baseline: boolean; cv: Record<string, { rmse: number; r2: number }>; band_pp_per_decade: number; caveats: string[]
  }
  importance: { feature: string; label: string; importance: number; direction: number }[]
  state: Record<string, { share: number; population?: number; migrants?: number; climate_effect_pp?: number }>
  districts: Dist[]
}

const card = 'rounded-xl border border-black/10 bg-white p-4'
const f1 = (n: number) => n.toFixed(1)
const big = (n: number) => (n >= 1e7 ? `${(n / 1e7).toFixed(2)} Cr` : n >= 1e5 ? `${(n / 1e5).toFixed(1)} L` : `${Math.round(n)}`)

export default function MigrationForecast() {
  const [d, setD] = useState<Forecast | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [year, setYear] = useState('2030')
  const [sort, setSort] = useState<'share' | 'effect' | 'change'>('share')

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/mh_forecast.json`)
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then(setD).catch((e) => setErr(String(e)))
  }, [])

  const rows = useMemo(() => {
    if (!d) return []
    const r = d.districts.map((x) => ({ ...x, p: x.pred[year] }))
    const key = { share: (x: typeof r[0]) => x.p.share, effect: (x: typeof r[0]) => x.p.climate_effect_pp, change: (x: typeof r[0]) => x.p.share - x.share_2011 }[sort]
    return r.sort((a, b) => key(b) - key(a))
  }, [d, year, sort])

  if (err) {
    return (
      <div className="mt-6 max-w-xl rounded-xl border border-black/10 bg-white p-5">
        <h3 className="text-base font-semibold">Forecast not generated yet</h3>
        <pre className="mt-3 rounded-lg bg-black/5 p-3 text-sm">{`python ml/mh_analysis.py\npython ml/mh_predict.py`}</pre>
        <p className="mt-2 text-xs opacity-60">({err})</p>
      </div>
    )
  }
  if (!d) return <div className="mt-6 text-sm opacity-70">Loading forecast…</div>

  const m = d.meta
  const base = m.cv['Average only (baseline)']
  const best = m.cv[m.best_model]
  const line = [{ year: 2011, share: d.state['2011'].share, effect: 0 },
    ...m.years.map((y) => ({ year: Number(y), share: d.state[y].share, effect: d.state[y].climate_effect_pp ?? 0 }))]
  const s = d.state[year]
  const maxImp = Math.max(...d.importance.map((i) => i.importance))
  const top = rows.slice(0, 15)

  return (
    <div className="mt-6 space-y-5">
      <div className="rounded-xl border border-amber-500/40 bg-amber-50 p-4 text-sm">
        <strong>Read this first.</strong> The model learned from {m.n_districts} districts and two Census years (2001, 2011) only.
        It is a climate-migration <em>scenario</em> with a wide band, not an exact forecast. {m.beats_baseline
          ? 'Leave-one-out testing shows it does beat a plain average, but only modestly.'
          : 'In leave-one-out testing it did NOT beat a plain average, so treat the numbers as illustration only.'}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm opacity-70">Year</span>
        {m.years.map((y) => (
          <button key={y} onClick={() => setYear(y)}
            className={`rounded-full border px-3 py-1 text-sm ${year === y ? 'border-[#1A2418] bg-[#1A2418] text-white' : 'border-black/15 bg-white'}`}>{y}</button>
        ))}
      </div>

      <div className="grid gap-3 md:grid-cols-4">
        <div className={card}><div className="text-xs opacity-60">Migrants from outside district, 2011 (Census)</div><div className="mt-1 text-xl font-semibold">{f1(d.state['2011'].share)}%</div><div className="text-xs opacity-60">of population</div></div>
        <div className={card}><div className="text-xs opacity-60">Predicted {year}</div><div className="mt-1 text-xl font-semibold">{f1(s.share)}%</div><div className="text-xs opacity-60">≈ {big(s.migrants ?? 0)} of {big(s.population ?? 0)} people</div></div>
        <div className={card}><div className="text-xs opacity-60">Part due to climate alone</div><div className="mt-1 text-xl font-semibold">{(s.climate_effect_pp ?? 0) >= 0 ? '+' : ''}{(s.climate_effect_pp ?? 0).toFixed(2)} pp</div><div className="text-xs opacity-60">vs 2001–10 climate</div></div>
        <div className={card}><div className="text-xs opacity-60">Model ({m.best_model})</div><div className="mt-1 text-xl font-semibold">R² {best.r2.toFixed(2)}</div><div className="text-xs opacity-60">leave-one-out; error ±{best.rmse.toFixed(1)} pp</div></div>
      </div>

      <div className={card}>
        <div className="text-sm font-medium">Maharashtra: migrants from outside the district, % of population</div>
        <div className="mb-1 text-xs opacity-60">{m.target}. 2011 = Census; later years = model scenario.</div>
        <div className="mt-2 h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={line} margin={{ left: 0, right: 16, top: 8 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
              <XAxis dataKey="year" type="number" domain={[2011, 2050]} ticks={[2011, 2030, 2035, 2040, 2050]} />
              <YAxis unit="%" />
              <Tooltip formatter={(v) => Number(v).toFixed(2)} />
              <Legend />
              <Line type="monotone" dataKey="share" name="Predicted share (%)" stroke="#1A2418" strokeWidth={2} dot />
              <Line type="monotone" dataKey="effect" name="Climate-only effect (pp)" stroke="#C0392B" strokeWidth={2} dot />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className={card}>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <div className="text-sm font-medium">Top 15 districts, {year}</div>
            <div className="text-xs opacity-60">Bar = predicted share (%), thin line on the table below shows the full band.</div>
          </div>
          <div className="flex gap-2 text-xs">
            {([['share', 'predicted share'], ['effect', 'climate effect'], ['change', 'change vs 2011']] as const).map(([k, l]) => (
              <button key={k} onClick={() => setSort(k)} className={`rounded-full border px-3 py-1 ${sort === k ? 'border-[#1A2418] bg-[#1A2418] text-white' : 'border-black/15'}`}>{l}</button>
            ))}
          </div>
        </div>
        <div className="mt-2 h-96">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={top.map((r) => ({ name: r.name, share: r.p.share }))} layout="vertical" margin={{ left: 80, right: 16 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
              <XAxis type="number" unit="%" />
              <YAxis type="category" dataKey="name" width={150} tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => `${Number(v).toFixed(1)}%`} />
              <Bar dataKey="share" name="Predicted share">
                {top.map((r) => <Cell key={r.id} fill={r.p.climate_effect_pp > 0 ? '#C0392B' : '#3D7A5E'} />)}
              </Bar>
              <ReferenceLine x={s.share} stroke="#1A2418" strokeDasharray="4 4" />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="text-xs opacity-60">Red = climate pushes migration up in this district, green = down. Dashed line = Maharashtra average.</p>

        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-xs">
            <thead><tr className="text-left opacity-60"><th className="py-1 font-normal">District</th><th className="font-normal">2001 %</th><th className="font-normal">2011 %</th><th className="font-normal">Predicted {year} %</th><th className="font-normal">Band</th><th className="font-normal">Climate effect (pp)</th><th className="font-normal">Migrants ≈</th></tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.id} className="border-t border-black/5">
                  <td className="py-1">{r.name}</td><td>{f1(r.share_2001)}</td><td>{f1(r.share_2011)}</td>
                  <td className="font-medium">{f1(r.p.share)}</td><td>{f1(r.p.low)}–{f1(r.p.high)}</td>
                  <td style={{ color: r.p.climate_effect_pp > 0 ? '#C0392B' : '#3D7A5E' }}>{r.p.climate_effect_pp >= 0 ? '+' : ''}{r.p.climate_effect_pp.toFixed(2)}</td>
                  <td>{big(r.p.migrants)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid gap-3 md:grid-cols-2">
        <div className={card}>
          <div className="text-sm font-medium">What the model relies on</div>
          <div className="mb-2 text-xs opacity-60">Share of importance. “↑” = higher value goes with more in-migration, “↓” = with less (association, not proof of cause).</div>
          <div className="space-y-2">
            {d.importance.map((i) => (
              <div key={i.feature} className="flex items-center gap-2 text-xs">
                <span className="w-44 truncate">{i.label} {i.direction > 0 ? '↑' : '↓'}</span>
                <div className="h-2 flex-1 rounded-full bg-black/10"><div className="h-2 rounded-full bg-[#4A7C59]" style={{ width: `${(i.importance / maxImp) * 100}%` }} /></div>
                <span className="w-10 text-right">{(i.importance * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>
        <div className={card}>
          <div className="text-sm font-medium">How good is it? (leave-one-district-out)</div>
          <table className="mt-2 w-full text-xs">
            <thead><tr className="text-left opacity-60"><th className="font-normal">Model</th><th className="font-normal">Error (pp)</th><th className="font-normal">R²</th></tr></thead>
            <tbody>
              {Object.entries(m.cv).map(([n, v]) => (
                <tr key={n} className={`border-t border-black/5 ${n === m.best_model ? 'font-semibold' : ''}`}><td className="py-1">{n}</td><td>{v.rmse.toFixed(2)}</td><td>{v.r2.toFixed(2)}</td></tr>
              ))}
            </tbody>
          </table>
          <p className="mt-2 text-xs opacity-60">Lower error than “Average only” ({base.rmse.toFixed(2)} pp) means the model has some skill. Generated {m.generated}.</p>
        </div>
      </div>

      <ul className="list-disc space-y-1 pl-5 text-xs opacity-70">
        {m.caveats.map((c) => <li key={c}>{c}</li>)}
      </ul>
    </div>
  )
}
