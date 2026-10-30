import { useEffect, useMemo, useState } from 'react'
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, Cell,
} from 'recharts'
import { MigrationAnalysisTab, RiskScoreTab, ProjectionTab } from './MaharashtraAnalysis'

/* ── Types (match public/data/maharashtra.json written by ml/mh_build.py) ── */

interface Annual {
  year: number; tmean: number; tmax: number; tmin: number
  rain: number; monsoon_rain: number; monsoon_dep: number
  heatwave_days: number; heatwave_spells: number; tmax_peak: number
  heavy_days: number; vheavy_days: number; rx1: number; rx5: number
  max_dry_spell: number; monsoon_class: string
  soil_premonsoon?: number; dry_soil_pct?: number; wsi: number
  drought_districts?: number
}
interface Trend { slope_decade: number; p: number | null; first: number; last: number; change: number }
interface District {
  id: string; name: string; region: string; lat: number; lon: number; coastal: boolean
  annual: Annual[]
  monthly: { m: number; tmax: number; tmin: number; rain: number }[]
  trends: Record<string, Trend | null>
  drought_years: number
}
interface MigRow {
  id: string; name: string
  pop_2001: number; pop_2011: number; migrants_2001: number; migrants_2011: number
  share_2001: number; share_2011: number; share_change_pp: number
  pop_growth_pct: number; migrant_growth_pct: number
  drought_years_2001_10: number; wsi_2001_10: number; heatwave_days_2001_10: number
  [k: string]: number | string
}
interface Data {
  meta: {
    generated: string; period: [number, number]; n_districts: number
    climate_source: string; soil_wetness: boolean; trend_method: string
    metrics: Record<string, { label: string; unit: string }>
    census_source: string | null; migration_message: string | null
  }
  districts: District[]
  state: { annual: Annual[]; trends: Record<string, Trend | null> }
  disasters: { year: number; hazard: string; events: number; deaths: number; affected: number }[] | null
  migration: null | {
    rows: MigRow[]
    correlations: { x: string; y: string; rho: number; p: number; n: number }[]
    state: Record<string, number>
    optional_columns: string[]
  }
}

const PALETTE = ['#B42318', '#2F6F8F', '#3B7A4B', '#E8590C', '#7A4FB5', '#C99700', '#0F766E', '#9A3412']
const CLASS_COLOR: Record<string, string> = {
  'severe drought': '#B42318', drought: '#E8590C', normal: '#9CA3AF', excess: '#2F6F8F',
}
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const card = 'rounded-xl border border-black/10 bg-white p-4'

const f = (v: number | null | undefined, d = 1) => (v === null || v === undefined ? '–' : v.toFixed(d))
const sig = (p: number | null) =>
  p === null ? 'n/a' : p < 0.05 ? 'significant (p<0.05)' : p < 0.1 ? 'weak (p<0.1)' : 'not significant'
const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / (xs.length || 1)
const big = (n: number) =>
  n >= 1e7 ? `${(n / 1e7).toFixed(2)} Cr` : n >= 1e5 ? `${(n / 1e5).toFixed(1)} L` : n.toLocaleString()

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className={card}>
      <div className="text-xs opacity-60">{label}</div>
      <div className="mt-1 text-xl font-semibold">{value}</div>
      {sub && <div className="mt-0.5 text-xs opacity-60">{sub}</div>}
    </div>
  )
}

function Chart({ title, note, children }: { title: string; note?: string; children: React.ReactNode }) {
  return (
    <div className={card}>
      <div className="text-sm font-medium">{title}</div>
      {note && <div className="mb-1 text-xs opacity-60">{note}</div>}
      <div className="mt-2 h-56">
        <ResponsiveContainer width="100%" height="100%">{children as React.ReactElement}</ResponsiveContainer>
      </div>
    </div>
  )
}

/* ── Tab 1: climate data ─────────────────────────────────────── */

function ClimateTab({ data }: { data: Data }) {
  const [q, setQ] = useState('')
  const [sel, setSel] = useState('pune')
  const list = data.districts.filter((d) => d.name.toLowerCase().includes(q.trim().toLowerCase()))
  const d = data.districts.find((x) => x.id === sel) ?? data.districts[0]
  const a = d.annual
  const last = a[a.length - 1]
  const [y0, y1] = data.meta.period

  return (
    <div className="mt-6 grid gap-6 lg:grid-cols-[280px_1fr]">
      <div>
        <input
          value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search district"
          className="w-full rounded-xl border border-black/15 bg-white px-4 py-2.5 text-sm outline-none focus:border-[#3B7A4B]"
        />
        <div className="mt-3 max-h-[560px] overflow-y-auto rounded-xl border border-black/10 bg-white">
          {list.map((x) => (
            <button
              key={x.id} onClick={() => setSel(x.id)}
              className={`flex w-full items-center justify-between border-b border-black/5 px-4 py-2.5 text-left text-sm last:border-b-0 hover:bg-black/[0.03] ${
                d.id === x.id ? 'bg-black/[0.05] font-medium' : ''
              }`}
            >
              <span>{x.name}</span>
              <span className="text-[10px] opacity-50">{x.region}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="space-y-5">
        <div>
          <h2 className="text-xl font-semibold">{d.name}</h2>
          <p className="text-xs opacity-60">
            {d.region} · grid cell at {d.lat.toFixed(2)}°N, {d.lon.toFixed(2)}°E · {y0}–{y1}
          </p>
        </div>

        <div className="grid gap-3 sm:grid-cols-3 xl:grid-cols-6">
          <Stat label={`Mean temp ${last.year}`} value={`${f(last.tmean)} °C`} sub={`${y0}–${y1} avg ${f(mean(a.map((r) => r.tmean)))}`} />
          <Stat label={`Rainfall ${last.year}`} value={`${f(last.rain, 0)} mm`} sub={`avg ${f(mean(a.map((r) => r.rain)), 0)} mm`} />
          <Stat label={`Heatwave days ${last.year}`} value={`${last.heatwave_days}`} sub={`avg ${f(mean(a.map((r) => r.heatwave_days)))}/yr`} />
          <Stat label={`Heavy-rain days ${last.year}`} value={`${last.heavy_days}`} sub={`≥64.5 mm/day · avg ${f(mean(a.map((r) => r.heavy_days)))}`} />
          <Stat label="Drought years" value={`${d.drought_years} of ${a.length}`} sub="monsoon ≤ −25% of mean" />
          <Stat label={`Water-stress ${last.year}`} value={`${f(last.wsi, 0)}/100`} sub={`avg ${f(mean(a.map((r) => r.wsi)), 0)}`} />
        </div>

        <div className="grid gap-4 xl:grid-cols-2">
          <Chart title="Temperature (°C)">
            <LineChart data={a}>
              <CartesianGrid strokeDasharray="3 3" stroke="#0001" />
              <XAxis dataKey="year" tick={{ fontSize: 10 }} /><YAxis domain={['auto', 'auto']} tick={{ fontSize: 10 }} width={34} />
              <Tooltip /><Legend wrapperStyle={{ fontSize: 11 }} />
              <Line dataKey="tmax" name="Mean max" stroke="#B42318" dot={false} />
              <Line dataKey="tmean" name="Mean" stroke="#E8590C" dot={false} />
              <Line dataKey="tmin" name="Mean min" stroke="#2F6F8F" dot={false} />
            </LineChart>
          </Chart>
          <Chart title="Rainfall (mm)">
            <BarChart data={a}>
              <CartesianGrid strokeDasharray="3 3" stroke="#0001" />
              <XAxis dataKey="year" tick={{ fontSize: 10 }} /><YAxis tick={{ fontSize: 10 }} width={40} />
              <Tooltip /><Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="monsoon_rain" name="Monsoon (Jun–Sep)" fill="#2F6F8F" />
              <Bar dataKey="rain" name="Annual" fill="#9CC3D5" />
            </BarChart>
          </Chart>
          <Chart title="Heatwave days per year" note="IMD-style: Tmax ≥40 °C (37 °C coastal) and ≥4.5 °C above normal, spells of 2+ days">
            <BarChart data={a}>
              <CartesianGrid strokeDasharray="3 3" stroke="#0001" />
              <XAxis dataKey="year" tick={{ fontSize: 10 }} /><YAxis allowDecimals={false} tick={{ fontSize: 10 }} width={30} />
              <Tooltip />
              <Bar dataKey="heatwave_days" name="Heatwave days" fill="#B42318" />
            </BarChart>
          </Chart>
          <Chart title="Flood proxy: heavy-rain days" note="Days with ≥64.5 mm (heavy) and ≥115.6 mm (very heavy) of rain">
            <BarChart data={a}>
              <CartesianGrid strokeDasharray="3 3" stroke="#0001" />
              <XAxis dataKey="year" tick={{ fontSize: 10 }} /><YAxis allowDecimals={false} tick={{ fontSize: 10 }} width={30} />
              <Tooltip /><Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="heavy_days" name="≥64.5 mm" fill="#2F6F8F" />
              <Bar dataKey="vheavy_days" name="≥115.6 mm" fill="#0F3F5C" />
            </BarChart>
          </Chart>
          <Chart title="Drought: monsoon rain vs district mean (%)" note="Red/orange = IMD meteorological drought (≤ −25%)">
            <BarChart data={a}>
              <CartesianGrid strokeDasharray="3 3" stroke="#0001" />
              <XAxis dataKey="year" tick={{ fontSize: 10 }} /><YAxis tick={{ fontSize: 10 }} width={36} />
              <Tooltip formatter={(v) => `${Number(v).toFixed(0)}%`} />
              <Bar dataKey="monsoon_dep" name="Departure %">
                {a.map((r) => <Cell key={r.year} fill={CLASS_COLOR[r.monsoon_class]} />)}
              </Bar>
            </BarChart>
          </Chart>
          <Chart
            title="Water stress"
            note={data.meta.soil_wetness
              ? 'Composite index 0–100 (low monsoon rain + dry root-zone soil + high Tmax) and % of days with dry soil'
              : 'Composite index 0–100 (low monsoon rain + high Tmax); soil wetness was not available'}
          >
            <LineChart data={a}>
              <CartesianGrid strokeDasharray="3 3" stroke="#0001" />
              <XAxis dataKey="year" tick={{ fontSize: 10 }} /><YAxis tick={{ fontSize: 10 }} width={30} />
              <Tooltip /><Legend wrapperStyle={{ fontSize: 11 }} />
              <Line dataKey="wsi" name="Water-stress index" stroke="#7A4FB5" dot={false} />
              {data.meta.soil_wetness && <Line dataKey="dry_soil_pct" name="% days dry soil" stroke="#C99700" dot={false} />}
            </LineChart>
          </Chart>
        </div>

        <Chart title="Monthly climate (average of all years)">
          <BarChart data={d.monthly.map((m) => ({ ...m, name: MONTHS[m.m - 1] }))}>
            <CartesianGrid strokeDasharray="3 3" stroke="#0001" />
            <XAxis dataKey="name" tick={{ fontSize: 10 }} />
            <YAxis yAxisId="r" tick={{ fontSize: 10 }} width={40} />
            <YAxis yAxisId="t" orientation="right" tick={{ fontSize: 10 }} width={30} />
            <Tooltip /><Legend wrapperStyle={{ fontSize: 11 }} />
            <Bar yAxisId="r" dataKey="rain" name="Rain (mm)" fill="#2F6F8F" />
            <Bar yAxisId="t" dataKey="tmax" name="Mean max (°C)" fill="#B42318" />
          </BarChart>
        </Chart>

        <div className={`${card} overflow-x-auto`}>
          <div className="mb-2 text-sm font-medium">Year-by-year values</div>
          <table className="w-full text-xs">
            <thead>
              <tr className="text-left opacity-60">
                {['Year', 'Mean °C', 'Rain mm', 'Monsoon mm', 'Dep %', 'Heatwave d', 'Heavy d', 'Dry spell d', 'Monsoon', 'WSI'].map((h) => (
                  <th key={h} className="py-1 pr-3 font-normal">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[...a].reverse().map((r) => (
                <tr key={r.year} className="border-t border-black/5">
                  <td className="py-1 pr-3">{r.year}</td><td>{f(r.tmean)}</td><td>{f(r.rain, 0)}</td>
                  <td>{f(r.monsoon_rain, 0)}</td><td>{f(r.monsoon_dep, 0)}</td><td>{r.heatwave_days}</td>
                  <td>{r.heavy_days}</td><td>{r.max_dry_spell}</td>
                  <td style={{ color: CLASS_COLOR[r.monsoon_class] }}>{r.monsoon_class}</td><td>{f(r.wsi, 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {data.disasters && (
          <div className={card}>
            <div className="mb-1 text-sm font-medium">Recorded disasters in Maharashtra (EM-DAT, state level)</div>
            <p className="mb-2 text-xs opacity-60">Real event records, not district-specific. One event touching several states is split equally.</p>
            <table className="w-full text-xs">
              <thead><tr className="text-left opacity-60"><th className="py-1 font-normal">Year</th><th className="font-normal">Hazard</th><th className="font-normal">Events</th><th className="font-normal">Deaths</th><th className="font-normal">Affected</th></tr></thead>
              <tbody>
                {[...data.disasters].sort((x, y) => y.year - x.year).map((e, i) => (
                  <tr key={i} className="border-t border-black/5"><td className="py-1">{e.year}</td><td>{e.hazard}</td><td>{f(e.events, 1)}</td><td>{Math.round(e.deaths)}</td><td>{big(e.affected)}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

/* ── Tab 2: trend analysis ───────────────────────────────────── */

function TrendTab({ data }: { data: Data }) {
  const [metric, setMetric] = useState('tmean')
  const [picked, setPicked] = useState<string[]>(['pune', 'nagpur'])
  const info = data.meta.metrics[metric]
  const [y0, y1] = data.meta.period

  const toggle = (id: string) =>
    setPicked((p) => (p.includes(id) ? p.filter((x) => x !== id) : p.length >= 6 ? p : [...p, id]))

  const chart = useMemo(() => {
    const years = data.state.annual.map((r) => r.year)
    return years.map((y, i) => {
      const row: Record<string, number> = { year: y, state: (data.state.annual[i] as unknown as Record<string, number>)[metric] }
      picked.forEach((id) => {
        const dd = data.districts.find((x) => x.id === id)
        if (dd) row[id] = (dd.annual[i] as unknown as Record<string, number>)[metric]
      })
      return row
    })
  }, [data, metric, picked])

  const ranking = useMemo(
    () =>
      data.districts
        .map((d) => ({ d, t: d.trends[metric] }))
        .filter((x): x is { d: District; t: Trend } => x.t !== null && x.t !== undefined)
        .sort((a, b) => b.t.slope_decade - a.t.slope_decade),
    [data, metric],
  )
  const st = data.state.trends[metric]

  return (
    <div className="mt-6 space-y-5">
      <div className="flex flex-wrap items-center gap-3">
        <select
          value={metric} onChange={(e) => setMetric(e.target.value)}
          className="rounded-xl border border-black/15 bg-white px-3 py-2 text-sm"
        >
          {Object.entries(data.meta.metrics).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}
        </select>
        <span className="text-xs opacity-60">Pick up to 6 districts to compare (Pune and Nagpur are selected first)</span>
      </div>

      <div className={`${card} flex max-h-40 flex-wrap gap-x-4 gap-y-1.5 overflow-y-auto`}>
        {data.districts.map((d) => (
          <label key={d.id} className="flex cursor-pointer items-center gap-1.5 text-xs">
            <input type="checkbox" checked={picked.includes(d.id)} onChange={() => toggle(d.id)} />
            {d.name}
          </label>
        ))}
      </div>

      <Chart title={`${info.label} (${info.unit}), ${y0}–${y1}`} note="Dashed grey = Maharashtra average (all districts, equal weight)">
        <LineChart data={chart}>
          <CartesianGrid strokeDasharray="3 3" stroke="#0001" />
          <XAxis dataKey="year" tick={{ fontSize: 10 }} /><YAxis domain={['auto', 'auto']} tick={{ fontSize: 10 }} width={40} />
          <Tooltip formatter={(v) => Number(v).toFixed(2)} /><Legend wrapperStyle={{ fontSize: 11 }} />
          <Line dataKey="state" name="Maharashtra avg" stroke="#6B7280" strokeDasharray="5 4" dot={false} />
          {picked.map((id, i) => (
            <Line key={id} dataKey={id} name={data.districts.find((x) => x.id === id)?.name} stroke={PALETTE[i % PALETTE.length]} dot={false} strokeWidth={2} />
          ))}
        </LineChart>
      </Chart>

      {st && (
        <div className={card}>
          <span className="font-medium">Maharashtra average:</span>{' '}
          {st.slope_decade >= 0 ? '+' : ''}{f(st.slope_decade, 2)} {info.unit} per decade ({sig(st.p)}) ·{' '}
          2001–10 mean {f(st.first, 2)} → 2016–25 mean {f(st.last, 2)} ({st.change >= 0 ? '+' : ''}{f(st.change, 2)})
        </div>
      )}

      <div className={`${card} overflow-x-auto`}>
        <div className="mb-1 text-sm font-medium">All districts ranked by trend: {info.label}</div>
        <p className="mb-2 text-xs opacity-60">{data.meta.trend_method}. Only 25 years of data: treat “not significant” trends as unproven.</p>
        <table className="w-full text-xs">
          <thead>
            <tr className="text-left opacity-60">
              <th className="py-1 font-normal">District</th><th className="font-normal">Region</th>
              <th className="font-normal">Per decade</th><th className="font-normal">2001–10 avg</th>
              <th className="font-normal">2016–25 avg</th><th className="font-normal">Change</th><th className="font-normal">Evidence</th>
            </tr>
          </thead>
          <tbody>
            {ranking.map(({ d, t }) => (
              <tr key={d.id} className="border-t border-black/5">
                <td className="py-1 font-medium">{d.name}</td><td className="opacity-70">{d.region}</td>
                <td>{t.slope_decade >= 0 ? '+' : ''}{f(t.slope_decade, 2)}</td>
                <td>{f(t.first, 2)}</td><td>{f(t.last, 2)}</td>
                <td>{t.change >= 0 ? '+' : ''}{f(t.change, 2)}</td>
                <td style={{ color: t.p !== null && t.p < 0.05 ? '#B42318' : undefined }}>{sig(t.p)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

/* ── Tab 3: historical migration ─────────────────────────────── */

function MigrationTab({ data }: { data: Data }) {
  const mig = data.migration
  const [sortKey, setSortKey] = useState<'share_2011' | 'share_change_pp' | 'pop_growth_pct'>('share_change_pp')

  if (!mig) {
    return (
      <div className={`${card} mt-6 max-w-2xl`}>
        <h3 className="text-base font-semibold">Census migration data not loaded yet</h3>
        <p className="mt-2 text-sm opacity-80">
          {data.meta.migration_message}. This page only shows real Census of India 2001 and 2011 figures, so it stays empty until you add them:
        </p>
        <pre className="mt-3 overflow-x-auto rounded-lg bg-black/5 p-3 text-xs">
{`1. cp data/census_template/migration_district_TEMPLATE.csv data/raw/census/migration_district.csv
2. Fill one row per district per census year (2001, 2011) from the Census D-series
   migration tables (D-1 / D-2) on censusindia.gov.in
3. python ml/mh_build.py`}
        </pre>
      </div>
    )
  }

  const rows = [...mig.rows].sort((a, b) => (b[sortKey] as number) - (a[sortKey] as number))
  const s = mig.state
  const chart = rows.map((r) => ({ name: r.name, '2001': r.share_2001, '2011': r.share_2011 }))

  return (
    <div className="mt-6 space-y-5">
      <div className="grid gap-3 sm:grid-cols-4">
        <Stat label="Migrants 2001" value={big(s.migrants_2001)} sub={`${f(s.share_2001)}% of ${big(s.pop_2001)} people`} />
        <Stat label="Migrants 2011" value={big(s.migrants_2011)} sub={`${f(s.share_2011)}% of ${big(s.pop_2011)} people`} />
        <Stat label="Change in share" value={`${s.share_2011 - s.share_2001 >= 0 ? '+' : ''}${f(s.share_2011 - s.share_2001, 2)} pp`} sub="2001 → 2011" />
        <Stat label="Districts" value={`${mig.rows.length}`} sub="Census districts (Palghar is inside Thane)" />
      </div>

      <div className={card}>
        <div className="text-sm font-medium">Migrants as % of district population</div>
        <p className="mb-1 text-xs opacity-60">Migrant = enumerated outside the place of last residence (Census definition). It measures people who moved <em>into</em> a district.</p>
        <div className="mt-2 h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chart}>
              <CartesianGrid strokeDasharray="3 3" stroke="#0001" />
              <XAxis dataKey="name" tick={{ fontSize: 9 }} interval={0} angle={-60} textAnchor="end" height={90} />
              <YAxis tick={{ fontSize: 10 }} width={34} />
              <Tooltip formatter={(v) => `${Number(v).toFixed(1)}%`} /><Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="2001" fill="#9CC3D5" /><Bar dataKey="2011" fill="#2F6F8F" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className={`${card} overflow-x-auto`}>
        <div className="mb-2 flex flex-wrap items-center gap-2 text-sm">
          <span className="font-medium">District table, sort by</span>
          {([['share_change_pp', 'change in share'], ['share_2011', 'share 2011'], ['pop_growth_pct', 'population growth']] as const).map(([k, l]) => (
            <button key={k} onClick={() => setSortKey(k)} className={`rounded-full border px-3 py-1 text-xs ${sortKey === k ? 'bg-[#1A2418] text-white' : 'border-black/15'}`}>{l}</button>
          ))}
        </div>
        <table className="w-full text-xs">
          <thead>
            <tr className="text-left opacity-60">
              {['District', 'Pop 2001', 'Pop 2011', 'Growth %', 'Migrants 2001', 'Migrants 2011', 'Share 2001', 'Share 2011', 'Δ pp', 'Drought yrs 01–10'].map((h) => <th key={h} className="py-1 pr-3 font-normal">{h}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} className="border-t border-black/5">
                <td className="py-1 pr-3 font-medium">{r.name}</td>
                <td>{big(r.pop_2001)}</td><td>{big(r.pop_2011)}</td><td>{f(r.pop_growth_pct)}</td>
                <td>{big(r.migrants_2001)}</td><td>{big(r.migrants_2011)}</td>
                <td>{f(r.share_2001)}</td><td>{f(r.share_2011)}</td>
                <td>{r.share_change_pp >= 0 ? '+' : ''}{f(r.share_change_pp, 2)}</td><td>{r.drought_years_2001_10}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {mig.correlations.length > 0 && (
        <div className={card}>
          <div className="mb-1 text-sm font-medium">Do climate-stressed districts show different migration? (exploratory)</div>
          <p className="mb-2 text-xs opacity-60">Spearman rank correlation across districts. Climate = NASA POWER 2001–2010, migration = Census change 2001→2011.</p>
          <table className="w-full text-xs">
            <thead><tr className="text-left opacity-60"><th className="py-1 font-normal">Climate indicator</th><th className="font-normal">Migration measure</th><th className="font-normal">ρ</th><th className="font-normal">p</th><th className="font-normal">n</th></tr></thead>
            <tbody>
              {mig.correlations.map((c, i) => (
                <tr key={i} className="border-t border-black/5">
                  <td className="py-1">{c.x}</td><td>{c.y}</td><td>{f(c.rho, 2)}</td>
                  <td style={{ color: c.p < 0.05 ? '#B42318' : undefined }}>{f(c.p, 3)}</td><td>{c.n}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-3 text-xs opacity-70">
            Read with care: this is correlation, not causation. Census “last residence” counts people arriving in a district, so climate-driven
            <em> out</em>-migration is only visible indirectly (slower population growth). Migration for marriage, work and education dominates the totals.
            Palghar is merged into Thane because it was created in 2014.
          </p>
        </div>
      )}
    </div>
  )
}

/* ── Page ────────────────────────────────────────────────────── */

type Tab = 'climate' | 'trend' | 'migration' | 'district' | 'risk' | 'future'

export default function MaharashtraClimate() {
  const [data, setData] = useState<Data | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [tab, setTab] = useState<Tab>('climate')

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/maharashtra.json`)
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then(setData)
      .catch((e) => setErr(String(e)))
  }, [])

  if (err) {
    return (
      <div className="max-w-xl p-8">
        <h1 className="text-xl font-semibold">Maharashtra data not found</h1>
        <p className="mt-2 text-sm opacity-80">Generate it first, from the project root:</p>
        <pre className="mt-3 rounded-lg bg-black/5 p-3 text-sm">{`python ml/mh_fetch.py\npython ml/mh_build.py`}</pre>
        <p className="mt-2 text-xs opacity-60">({err})</p>
      </div>
    )
  }
  if (!data) return <div className="p-8 text-sm opacity-70">Loading Maharashtra data…</div>

  const tabs: { id: Tab; label: string }[] = [
    { id: 'climate', label: 'Climate data' },
    { id: 'trend', label: 'Trend analysis' },
    { id: 'migration', label: 'Historical migration' },
    { id: 'district', label: 'District migration' },
    { id: 'risk', label: 'Climate risk score' },
    { id: 'future', label: 'Future projections' },
  ]

  return (
    <div className="min-h-screen bg-[#F4F3EC] px-5 py-8 md:px-10" style={{ color: '#1A2418' }}>
      <div className="max-w-7xl">
        <h1 className="text-2xl font-semibold md:text-3xl">Maharashtra · Climate & Migration</h1>
        <p className="mt-1 text-sm opacity-70">
          {data.meta.n_districts} districts · {data.meta.period[0]}–{data.meta.period[1]} · {data.meta.climate_source}
        </p>
        <div className="mt-5 flex flex-wrap gap-2">
          {tabs.map((t) => (
            <button
              key={t.id} onClick={() => setTab(t.id)}
              className={`rounded-full border px-4 py-1.5 text-sm ${tab === t.id ? 'border-[#1A2418] bg-[#1A2418] text-white' : 'border-black/15 bg-white'}`}
            >
              {t.label}
            </button>
          ))}
        </div>

        {tab === 'climate' && <ClimateTab data={data} />}
        {tab === 'trend' && <TrendTab data={data} />}
        {tab === 'migration' && <MigrationTab data={data} />}
        {tab === 'district' && <MigrationAnalysisTab />}
        {tab === 'risk' && <RiskScoreTab />}
        {tab === 'future' && <ProjectionTab />}

        <p className="mt-10 max-w-4xl text-xs opacity-60">
          Limits: NASA POWER is a ~50 km reanalysis grid, so it smooths local extremes (fewer heavy-rain and heatwave days than a rain gauge
          would record) and nearby districts share similar values. Indicators follow IMD thresholds but are not official IMD declarations.
          Trends use 25 years only. Research prototype, not an official warning.
        </p>
      </div>
    </div>
  )
}
