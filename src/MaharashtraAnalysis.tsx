import { useEffect, useMemo, useState } from 'react'
import {
  ComposedChart, Line, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, Cell, ReferenceLine,
} from 'recharts'

/* Maharashtra tabs 5-7: district migration, climate-migration risk score, future projections.
   Reads public/data/mh_analysis.json, produced by `python ml/mh_analysis.py`.
   Used only by MaharashtraClimate.tsx - the India page does not import anything from here. */

/* ── Types (match ml/mh_analysis.py) ─────────────────────────── */

type Num = number | null
interface Band { mid: number; low: number; high: number }
interface ProjMetric {
  baseline: number; slope_decade: number; p: number | null; applied: boolean
  history: { year: number; v: number }[]; values: Record<string, Band>
}
interface FutureRisk { score: Num; level: string; factors: Record<string, Num> }
interface DistrictA {
  id: string; name: string; region: string; census_id: string
  risk: { inputs: Record<string, Num>; factors: Record<string, Num>; score: number; level: string; rank: number }
  future_risk: Record<string, FutureRisk>
  projection: Record<string, ProjMetric>
}
interface MigRowA {
  id: string; name: string
  pop_2001: number; pop_2011: number; migrants_2001: number; migrants_2011: number
  inflow_share_2001: number; inflow_share_2011: number; inflow_change: number; inflow_share_change_pp: number
  pop_growth_pct: number; net_vs_state_pp: number; net_vs_state_persons: number; movement: string
  risk_score: Num; risk_level: string | null
  out_2001?: number; out_2011?: number; out_share_2011?: number; net_flow_2011?: number
  [k: string]: number | string | null | undefined
}
interface Analysis {
  meta: {
    generated: string; baseline: [number, number]; future_years: number[]
    weights: Record<string, number>; factor_labels: Record<string, string>
    indicator_labels: Record<string, string>; factor_indicators: Record<string, string[]>
    levels: string[]; level_cutoffs: number[]; p_apply: number
    proj_metrics: Record<string, { label: string; unit: string }>
    migration_message: string | null
  }
  migration: null | {
    rows: MigRowA[]
    state: Record<string, number>
    correlations: { x: string; y: string; rho: number; p: number; n: number }[]
    has_outflow: boolean; streams: string[]; near_state_pp: number
  }
  risk_summary: { levels: Record<string, Record<string, number>>; mean_score: Record<string, number> }
  districts: DistrictA[]
  state_projection: Record<string, { history: { year: number; v: number }[]; values: Record<string, Band>; applied_districts: number; n: number }>
}

/* ── shared bits ─────────────────────────────────────────────── */

const card = 'rounded-xl border border-black/10 bg-white p-4'
const LEVEL_COLOR: Record<string, string> = {
  Low: '#3B7A4B', Moderate: '#C99700', High: '#E8590C', 'Very High': '#B42318', 'n/a': '#9CA3AF',
}
const FACTOR_COLOR: Record<string, string> = {
  heat: '#B42318', drought: '#E8590C', flood: '#2F6F8F', rainfall: '#0F766E', water: '#7A4FB5',
}
const STREAM_LABEL: Record<string, string> = {
  intra_district: 'Same district', inter_district: 'Other district, Maharashtra',
  inter_state: 'Other state', outside_india: 'Outside India',
}
const STREAM_COLOR: Record<string, string> = {
  intra_district: '#9CC3D5', inter_district: '#2F6F8F', inter_state: '#E8590C', outside_india: '#7A4FB5',
}

const f = (v: number | null | undefined, d = 1) => (v === null || v === undefined || !Number.isFinite(v) ? '–' : v.toFixed(d))
const sgn = (v: number, d = 1) => `${v >= 0 ? '+' : ''}${v.toFixed(d)}`
const big = (n: number) =>
  Math.abs(n) >= 1e7 ? `${(n / 1e7).toFixed(2)} Cr` : Math.abs(n) >= 1e5 ? `${(n / 1e5).toFixed(1)} L` : Math.round(n).toLocaleString()
const sigText = (p: number | null) =>
  p === null ? 'n/a' : p < 0.05 ? 'significant (p<0.05)' : p < 0.1 ? 'weak (p<0.1)' : 'not significant'

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className={card}>
      <div className="text-xs opacity-60">{label}</div>
      <div className="mt-1 text-xl font-semibold">{value}</div>
      {sub && <div className="mt-0.5 text-xs opacity-60">{sub}</div>}
    </div>
  )
}

function Pill({ level }: { level: string }) {
  return (
    <span className="inline-block whitespace-nowrap rounded-full px-2 py-0.5 text-[11px] font-medium text-white" style={{ background: LEVEL_COLOR[level] ?? '#9CA3AF' }}>
      {level}
    </span>
  )
}

function Box({ title, note, h = 'h-64', children }: { title: string; note?: string; h?: string; children: React.ReactNode }) {
  return (
    <div className={card}>
      <div className="text-sm font-medium">{title}</div>
      {note && <div className="mb-1 text-xs opacity-60">{note}</div>}
      <div className={`mt-2 ${h}`}>
        <ResponsiveContainer width="100%" height="100%">{children as React.ReactElement}</ResponsiveContainer>
      </div>
    </div>
  )
}

/* one fetch for all three tabs */
let cache: Promise<Analysis> | null = null
function loadAnalysis(): Promise<Analysis> {
  if (!cache) {
    cache = fetch(`${import.meta.env.BASE_URL}data/mh_analysis.json`)
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() as Promise<Analysis> })
      .catch((e) => { cache = null; throw e })
  }
  return cache
}

function Gate({ children }: { children: (d: Analysis) => React.ReactNode }) {
  const [data, setData] = useState<Analysis | null>(null)
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => {
    loadAnalysis().then(setData).catch((e) => setErr(String(e)))
  }, [])
  if (err) {
    return (
      <div className={`${card} mt-6 max-w-xl`}>
        <h3 className="text-base font-semibold">Analysis data not found</h3>
        <p className="mt-2 text-sm opacity-80">Generate it first, from the project root:</p>
        <pre className="mt-3 rounded-lg bg-black/5 p-3 text-sm">{`python ml/mh_build.py\npython ml/mh_analysis.py`}</pre>
        <p className="mt-2 text-xs opacity-60">({err})</p>
      </div>
    )
  }
  if (!data) return <div className="mt-6 text-sm opacity-70">Loading analysis…</div>
  return <>{children(data)}</>
}

/* ═══ 5. District-wise migration analysis ═══════════════════════ */

type MigSort = 'net_vs_state_pp' | 'inflow_share_2011' | 'inflow_change' | 'pop_growth_pct'

function MigrationView({ d }: { d: Analysis }) {
  const mig = d.migration
  const [sortKey, setSortKey] = useState<MigSort>('net_vs_state_pp')
  const rows = useMemo(
    () => (mig ? [...mig.rows].sort((a, b) => (b[sortKey] as number) - (a[sortKey] as number)) : []),
    [mig, sortKey],
  )

  if (!mig) {
    return (
      <div className={`${card} mt-6 max-w-2xl`}>
        <h3 className="text-base font-semibold">District migration needs the Census data</h3>
        <p className="mt-2 text-sm opacity-80">
          {d.meta.migration_message}. This page shows only real Census of India 2001 and 2011 numbers, so it stays empty until the CSV is filled:
        </p>
        <pre className="mt-3 overflow-x-auto rounded-lg bg-black/5 p-3 text-xs">
{`1. Open  data/raw/census/migration_district.csv
2. One row per district per census year (2001 and 2011), from the Census D-series tables
   Required: district, year, total_population, total_migrants
   Optional: migrants_intra_district, migrants_inter_district, migrants_inter_state,
             migrants_outside_india, out_migrants
3. python ml/mh_build.py
4. python ml/mh_analysis.py`}
        </pre>
      </div>
    )
  }

  const s = mig.state
  const gaining = mig.rows.filter((r) => r.movement === 'Gaining').length
  const losing = mig.rows.filter((r) => r.movement === 'Losing').length
  const netChart = [...mig.rows].sort((a, b) => b.net_vs_state_pp - a.net_vs_state_pp).map((r) => ({ name: r.name, v: r.net_vs_state_pp }))
  const flowChart = mig.has_outflow
    ? [...mig.rows].sort((a, b) => (b.net_flow_2011 ?? 0) - (a.net_flow_2011 ?? 0)).map((r) => ({
        name: r.name, Inflow: r.inflow_share_2011, Outflow: r.out_share_2011 ?? 0,
      }))
    : []
  const streamChart = mig.streams.length
    ? mig.rows.map((r) => ({ name: r.name, ...Object.fromEntries(mig.streams.map((k) => [k, (r[`${k}_pct_2011`] as number) ?? 0])) }))
    : []

  return (
    <div className="mt-6 space-y-5">
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <Stat label="Inflow: migrants 2011" value={big(s.migrants_2011)} sub={`${f(s.inflow_share_2011)}% of ${big(s.pop_2011)} (2001: ${f(s.inflow_share_2001)}%)`} />
        <Stat
          label={mig.has_outflow ? 'Outflow: out-migrants 2011' : 'Outflow'}
          value={mig.has_outflow ? big(s.out_2011) : 'not published'}
          sub={mig.has_outflow ? `net flow ${big(s.migrants_2011 - s.out_2011)}` : 'Census gives no district out-migration; see proxy below'}
        />
        <Stat label="Population growth 2001→11" value={`${f(s.growth_pct)}%`} sub="Maharashtra total, the yardstick for the proxy" />
        <Stat label="Gaining / losing vs state" value={`${gaining} / ${losing}`} sub={`beyond ±${mig.near_state_pp} pp of state growth`} />
      </div>

      <Box
        title="Net movement vs Maharashtra: population growth 2001→11 minus state growth (pp)"
        note="Proxy, not a count of movers. Green = district grew faster than the state (gaining people relative to the rest), red = slower (losing). Births and deaths differ by district, so this shows direction, not exact numbers."
        h="h-80"
      >
        <BarChart data={netChart}>
          <CartesianGrid strokeDasharray="3 3" stroke="#0001" />
          <XAxis dataKey="name" tick={{ fontSize: 9 }} interval={0} angle={-60} textAnchor="end" height={90} />
          <YAxis tick={{ fontSize: 10 }} width={34} />
          <ReferenceLine y={0} stroke="#0006" />
          <Tooltip formatter={(v) => `${sgn(Number(v))} pp`} />
          <Bar dataKey="v" name="vs state (pp)">
            {netChart.map((r) => <Cell key={r.name} fill={r.v >= 0 ? '#3B7A4B' : '#B42318'} />)}
          </Bar>
        </BarChart>
      </Box>

      {mig.has_outflow && (
        <Box title="Inflow vs outflow in 2011 (% of district population)" note="Inflow = Census migrants enumerated in the district; outflow = your out_migrants column. Sorted by net flow." h="h-80">
          <BarChart data={flowChart}>
            <CartesianGrid strokeDasharray="3 3" stroke="#0001" />
            <XAxis dataKey="name" tick={{ fontSize: 9 }} interval={0} angle={-60} textAnchor="end" height={90} />
            <YAxis tick={{ fontSize: 10 }} width={34} />
            <Tooltip formatter={(v) => `${Number(v).toFixed(1)}%`} /><Legend wrapperStyle={{ fontSize: 11 }} />
            <Bar dataKey="Inflow" fill="#2F6F8F" /><Bar dataKey="Outflow" fill="#E8590C" />
          </BarChart>
        </Box>
      )}

      {streamChart.length > 0 && (
        <Box title="Where do the 2011 migrants come from? (% of each district's migrants)" h="h-80">
          <BarChart data={streamChart}>
            <CartesianGrid strokeDasharray="3 3" stroke="#0001" />
            <XAxis dataKey="name" tick={{ fontSize: 9 }} interval={0} angle={-60} textAnchor="end" height={90} />
            <YAxis tick={{ fontSize: 10 }} width={34} domain={[0, 100]} />
            <Tooltip formatter={(v) => `${Number(v).toFixed(1)}%`} /><Legend wrapperStyle={{ fontSize: 11 }} />
            {mig.streams.map((k) => <Bar key={k} dataKey={k} name={STREAM_LABEL[k]} stackId="a" fill={STREAM_COLOR[k]} />)}
          </BarChart>
        </Box>
      )}

      <div className={`${card} overflow-x-auto`}>
        <div className="mb-2 flex flex-wrap items-center gap-2 text-sm">
          <span className="font-medium">District table, sort by</span>
          {([['net_vs_state_pp', 'net vs state'], ['inflow_share_2011', 'inflow share 2011'], ['inflow_change', 'change in migrants'], ['pop_growth_pct', 'population growth']] as const).map(([k, l]) => (
            <button key={k} onClick={() => setSortKey(k)} className={`rounded-full border px-3 py-1 text-xs ${sortKey === k ? 'bg-[#1A2418] text-white' : 'border-black/15'}`}>{l}</button>
          ))}
        </div>
        <table className="w-full text-xs">
          <thead>
            <tr className="text-left opacity-60">
              {['District', 'Pop 2011', 'Growth %', 'Migrants 2001', 'Migrants 2011', 'Share 2011', ...(mig.has_outflow ? ['Out-migrants 2011', 'Net flow'] : []), 'vs state (pp)', 'Movement', 'Climate risk'].map((h) => (
                <th key={h} className="py-1 pr-3 font-normal">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} className="border-t border-black/5">
                <td className="py-1 pr-3 font-medium">{r.name}</td>
                <td>{big(r.pop_2011)}</td><td>{f(r.pop_growth_pct)}</td>
                <td>{big(r.migrants_2001)}</td><td>{big(r.migrants_2011)}</td><td>{f(r.inflow_share_2011)}%</td>
                {mig.has_outflow && <><td>{big(r.out_2011 ?? 0)}</td><td>{big(r.net_flow_2011 ?? 0)}</td></>}
                <td style={{ color: r.net_vs_state_pp >= 0 ? '#3B7A4B' : '#B42318' }}>{sgn(r.net_vs_state_pp, 1)}</td>
                <td>{r.movement}</td>
                <td>{r.risk_level ? <Pill level={r.risk_level} /> : '–'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {mig.correlations.length > 0 && (
        <div className={card}>
          <div className="mb-1 text-sm font-medium">Do higher-risk districts move differently? (exploratory)</div>
          <p className="mb-2 text-xs opacity-60">Spearman rank correlation across districts. Risk score = 2016–25 climate; migration = Census change 2001→2011 (the climate comes after the migration, so this is a pattern check, not evidence of cause).</p>
          <table className="w-full text-xs">
            <thead><tr className="text-left opacity-60"><th className="py-1 font-normal">Climate measure</th><th className="font-normal">Migration measure</th><th className="font-normal">ρ</th><th className="font-normal">p</th><th className="font-normal">n</th></tr></thead>
            <tbody>
              {mig.correlations.map((c, i) => (
                <tr key={i} className="border-t border-black/5">
                  <td className="py-1">{c.x}</td><td>{c.y}</td><td>{f(c.rho, 2)}</td>
                  <td style={{ color: c.p < 0.05 ? '#B42318' : undefined }}>{f(c.p, 3)}</td><td>{c.n}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <p className="max-w-4xl text-xs opacity-60">
        Census “migrant” = person enumerated outside the place of last residence, so these are stocks of people who arrived over their lifetime, not yearly flows.
        District out-migration is not tabulated by the Census; the proxy above only shows relative gain or loss. Palghar (created 2014) is inside Thane.
        Marriage, work and education dominate migration, climate is only one driver.
      </p>
    </div>
  )
}

/* ═══ 6. Climate migration risk score ═══════════════════════════ */

function RiskView({ d }: { d: Analysis }) {
  const { meta } = d
  const [q, setQ] = useState('')
  const [region, setRegion] = useState('All')
  const [level, setLevel] = useState('All')
  const [sel, setSel] = useState<string>(() => [...d.districts].sort((a, b) => a.risk.rank - b.risk.rank)[0].id)
  const regions = useMemo(() => ['All', ...Array.from(new Set(d.districts.map((x) => x.region))).sort()], [d])
  const ranked = useMemo(() => [...d.districts].sort((a, b) => a.risk.rank - b.risk.rank), [d])
  const list = ranked.filter((x) =>
    x.name.toLowerCase().includes(q.trim().toLowerCase()) &&
    (region === 'All' || x.region === region) && (level === 'All' || x.risk.level === level))
  const cur = d.districts.find((x) => x.id === sel) ?? ranked[0]
  const [b0, b1] = meta.baseline
  const counts = d.risk_summary.levels[String(b1)]
  const factors = Object.keys(meta.weights)
  const chart = ranked.map((x) => ({ name: x.name, score: x.risk.score, level: x.risk.level }))

  return (
    <div className="mt-6 space-y-5">
      <div className="grid gap-3 sm:grid-cols-4">
        {meta.levels.map((lv) => (
          <div key={lv} className={card} style={{ borderLeft: `4px solid ${LEVEL_COLOR[lv]}` }}>
            <div className="text-xs opacity-60">{lv} risk</div>
            <div className="mt-1 text-xl font-semibold">{counts[lv]} districts</div>
            <div className="mt-0.5 text-xs opacity-60">
              score {lv === 'Low' ? `< ${f(meta.level_cutoffs[0])}` : lv === 'Moderate' ? `${f(meta.level_cutoffs[0])}–${f(meta.level_cutoffs[1])}` : lv === 'High' ? `${f(meta.level_cutoffs[1])}–${f(meta.level_cutoffs[2])}` : `≥ ${f(meta.level_cutoffs[2])}`}
            </div>
          </div>
        ))}
      </div>

      <div className="grid gap-6 lg:grid-cols-[360px_1fr]">
        <div>
          <input
            value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search district"
            className="w-full rounded-xl border border-black/15 bg-white px-4 py-2.5 text-sm outline-none focus:border-[#3B7A4B]"
          />
          <div className="mt-2 flex gap-2">
            <select value={region} onChange={(e) => setRegion(e.target.value)} className="w-1/2 rounded-xl border border-black/15 bg-white px-2 py-2 text-xs">
              {regions.map((r) => <option key={r}>{r}</option>)}
            </select>
            <select value={level} onChange={(e) => setLevel(e.target.value)} className="w-1/2 rounded-xl border border-black/15 bg-white px-2 py-2 text-xs">
              {['All', ...meta.levels].map((r) => <option key={r}>{r}</option>)}
            </select>
          </div>
          <div className="mt-3 max-h-[620px] overflow-y-auto rounded-xl border border-black/10 bg-white">
            {list.map((x) => (
              <button
                key={x.id} onClick={() => setSel(x.id)}
                className={`flex w-full items-center gap-3 border-b border-black/5 px-3 py-2 text-left text-sm last:border-b-0 hover:bg-black/[0.03] ${cur.id === x.id ? 'bg-black/[0.05]' : ''}`}
              >
                <span className="w-6 text-xs opacity-50">{x.risk.rank}</span>
                <span className="min-w-0 flex-1">
                  <span className="block truncate font-medium">{x.name}</span>
                  <span className="block h-1.5 rounded-full bg-black/10">
                    <span className="block h-full rounded-full" style={{ width: `${x.risk.score}%`, background: LEVEL_COLOR[x.risk.level] }} />
                  </span>
                </span>
                <span className="w-9 text-right text-xs">{f(x.risk.score, 0)}</span>
                <Pill level={x.risk.level} />
              </button>
            ))}
            {list.length === 0 && <div className="p-4 text-sm opacity-60">No district matches.</div>}
          </div>
        </div>

        <div className="space-y-5">
          <div className={card}>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <h2 className="text-xl font-semibold">{cur.name}</h2>
                <p className="text-xs opacity-60">{cur.region} · rank {cur.risk.rank} of {d.districts.length} · based on {b0}–{b1}</p>
              </div>
              <div className="text-right">
                <div className="text-3xl font-semibold">{f(cur.risk.score, 0)}<span className="text-sm opacity-50">/100</span></div>
                <Pill level={cur.risk.level} />
              </div>
            </div>

            <div className="mt-4 space-y-2.5">
              {factors.map((k) => (
                <div key={k}>
                  <div className="flex justify-between text-xs">
                    <span>{meta.factor_labels[k]} <span className="opacity-50">· weight {Math.round(meta.weights[k] * 100)}%</span></span>
                    <span className="font-medium">{f(cur.risk.factors[k], 0)}</span>
                  </div>
                  <div className="mt-0.5 h-2 rounded-full bg-black/10">
                    <div className="h-full rounded-full" style={{ width: `${cur.risk.factors[k] ?? 0}%`, background: FACTOR_COLOR[k] }} />
                  </div>
                  <div className="mt-0.5 text-[11px] opacity-60">
                    {meta.factor_indicators[k].map((ind) => `${meta.indicator_labels[ind]}: ${f(cur.risk.inputs[ind], ind === 'soil_premonsoon' ? 2 : 1)}`).join('  ·  ')}
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-4 border-t border-black/10 pt-3">
              <div className="mb-1.5 text-xs opacity-60">Outlook (trend-based, see Future projections)</div>
              <div className="flex flex-wrap items-center gap-3 text-xs">
                <span>{b1}: <Pill level={cur.risk.level} /></span>
                {meta.future_years.map((y) => (
                  <span key={y}>{y}: <Pill level={cur.future_risk[String(y)].level} /></span>
                ))}
              </div>
            </div>
          </div>

          <Box title="Climate migration risk score, all districts" note="Percentile-based composite, 0–100. Colour = level." h="h-[760px]">
            <BarChart data={chart} layout="vertical" margin={{ left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#0001" />
              <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 10 }} />
              <YAxis type="category" dataKey="name" tick={{ fontSize: 9 }} width={150} interval={0} />
              <Tooltip formatter={(v) => `${Number(v).toFixed(1)}`} />
              <Bar dataKey="score" name="Risk score">
                {chart.map((r) => <Cell key={r.name} fill={LEVEL_COLOR[r.level]} />)}
              </Bar>
            </BarChart>
          </Box>
        </div>
      </div>

      <div className={`${card} text-xs`}>
        <div className="mb-1 text-sm font-medium">How the score is built</div>
        <ul className="list-disc space-y-1 pl-5 opacity-80">
          <li>Five factors (heat, drought, flood, rainfall, water stress), each the average of two real indicators from the NASA POWER data. Weights above are a modelling choice, not measured.</li>
          <li>Every indicator becomes a 0–100 score = its percentile among the {d.districts.length} districts ({b0}–{b1} values; drought frequency and rainfall variability use all 2001–25). Higher = worse.</li>
          <li>Levels are quartiles of the {b0}–{b1} composite, so they are <strong>relative to other Maharashtra districts</strong>: “Very High” means among the worst quarter, not an absolute danger level.</li>
          <li>This scores climate stress that can push people to move. It does not measure actual migration, adaptive capacity, irrigation or income. NASA POWER is a ~50 km grid, so neighbouring districts look similar and local extremes are smoothed.</li>
        </ul>
      </div>
    </div>
  )
}

/* ═══ 7. Future climate projections ═════════════════════════════ */

interface Row { year: number; actual?: number; proj?: number; band?: [number, number] }

function buildRows(history: { year: number; v: number }[], values: Record<string, Band>, years: number[]): Row[] {
  const rows: Row[] = history.map((h) => ({ year: h.year, actual: h.v }))
  const last = rows[rows.length - 1]
  last.proj = last.actual
  last.band = [last.actual as number, last.actual as number]
  years.forEach((y) => {
    const b = values[String(y)]
    rows.push({ year: y, proj: b.mid, band: [b.low, b.high] })
  })
  return rows
}

function ProjectionView({ d }: { d: Analysis }) {
  const { meta } = d
  const years = meta.future_years
  const [sel, setSel] = useState('state')
  const [metric, setMetric] = useState('tmax')
  const [b0, b1] = meta.baseline
  const ranked = useMemo(() => [...d.districts].sort((a, b) => a.name.localeCompare(b.name)), [d])
  const cur = d.districts.find((x) => x.id === sel) ?? null
  const info = meta.proj_metrics[metric]

  const series = cur ? cur.projection[metric] : undefined
  const stateS = d.state_projection[metric]
  const history = cur ? series?.history : stateS?.history
  const values = cur ? series?.values : stateS?.values
  const rows = history && values ? buildRows(history, values, years) : null
  const base = cur
    ? series?.baseline
    : stateS ? stateS.history.filter((h) => h.year >= b0 && h.year <= b1).reduce((a, h) => a + h.v, 0) / (b1 - b0 + 1) : undefined
  const d1 = (v: number) => (metric === 'soil_premonsoon' ? v.toFixed(3) : v.toFixed(1))

  const levelChart = [b1, ...years].map((y) => ({ year: String(y), ...d.risk_summary.levels[String(y)] }))
  const outlook = useMemo(() => [...d.districts].sort((a, b) => a.risk.rank - b.risk.rank), [d])

  return (
    <div className="mt-6 space-y-5">
      <div className={`${card} border-[#E8590C]/40 bg-[#FFF8F1] text-xs`}>
        <strong>Read this first.</strong> These are statistical trend extrapolations of the 2001–{b1} NASA POWER record, not climate-model (CMIP6/SSP) projections.
        A trend is applied only where it is statistically visible (p &lt; {meta.p_apply}); otherwise the value stays at the {b0}–{b1} average, and the band shows how wide the uncertainty is.
        This 25-year grid record shows no clear warming at many districts even though India is warming overall, so do not read a flat line as “no climate change”.
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <select value={sel} onChange={(e) => setSel(e.target.value)} className="rounded-xl border border-black/15 bg-white px-3 py-2 text-sm">
          <option value="state">Maharashtra average</option>
          {ranked.map((x) => <option key={x.id} value={x.id}>{x.name}</option>)}
        </select>
        <select value={metric} onChange={(e) => setMetric(e.target.value)} className="rounded-xl border border-black/15 bg-white px-3 py-2 text-sm">
          {Object.entries(meta.proj_metrics).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}
        </select>
      </div>

      {rows ? (
        <Box
          title={`${info.label} (${info.unit}): ${cur ? cur.name : 'Maharashtra average'}, history and projection`}
          note={cur && series
            ? `Trend ${sgn(series.slope_decade, 2)} ${info.unit} per decade, ${sigText(series.p)} → ${series.applied ? 'applied to the projection' : `not applied (projection = ${b0}–${b1} average)`}. Shaded = plausible range.`
            : `Average of ${stateS?.n} districts; the trend was applied in ${stateS?.applied_districts} of them. Shaded = plausible range.`}
          h="h-80"
        >
          <ComposedChart data={rows}>
            <CartesianGrid strokeDasharray="3 3" stroke="#0001" />
            <XAxis dataKey="year" type="number" domain={[2001, 2050]} ticks={[2001, 2010, 2020, 2025, 2030, 2035, 2040, 2050]} tick={{ fontSize: 10 }} />
            <YAxis domain={['auto', 'auto']} tick={{ fontSize: 10 }} width={44} tickFormatter={(v) => d1(Number(v))} />
            <Tooltip formatter={(v) => (Array.isArray(v) ? `${d1(Number(v[0]))} – ${d1(Number(v[1]))}` : d1(Number(v)))} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            {base !== undefined && <ReferenceLine y={base} stroke="#6B7280" strokeDasharray="4 4" label={{ value: `${b0}–${b1} avg`, fontSize: 10, fill: '#6B7280', position: 'insideTopLeft' }} />}
            <Area dataKey="band" name="Plausible range" stroke="none" fill="#E8590C" fillOpacity={0.15} connectNulls isAnimationActive={false} />
            <Line dataKey="actual" name="Observed" stroke="#1A2418" dot={false} strokeWidth={2} connectNulls isAnimationActive={false} />
            <Line dataKey="proj" name="Projection" stroke="#E8590C" strokeDasharray="5 4" strokeWidth={2} dot={{ r: 3 }} connectNulls isAnimationActive={false} />
          </ComposedChart>
        </Box>
      ) : (
        <div className={card}>
          <p className="text-sm">No projection for this indicator in {cur?.name}: the yearly series is too flat or has too few distinct values to fit a trend.</p>
        </div>
      )}

      {cur && (
        <div className={`${card} overflow-x-auto`}>
          <div className="mb-2 text-sm font-medium">{cur.name}: projected values (middle, with range)</div>
          <table className="w-full text-xs">
            <thead>
              <tr className="text-left opacity-60">
                <th className="py-1 font-normal">Indicator</th><th className="font-normal">{b0}–{b1} avg</th>
                {years.map((y) => <th key={y} className="font-normal">{y}</th>)}<th className="font-normal">Trend / decade</th><th className="font-normal">Evidence</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(meta.proj_metrics).map(([k, m]) => {
                const p = cur.projection[k]
                if (!p) return <tr key={k} className="border-t border-black/5"><td className="py-1">{m.label}</td><td colSpan={years.length + 3} className="opacity-50">no projection (series too flat)</td></tr>
                const dd = (v: number) => (k === 'soil_premonsoon' ? v.toFixed(3) : v.toFixed(1))
                return (
                  <tr key={k} className="border-t border-black/5">
                    <td className="py-1 pr-3 font-medium">{m.label} <span className="opacity-50">({m.unit})</span></td>
                    <td>{dd(p.baseline)}</td>
                    {years.map((y) => {
                      const b = p.values[String(y)]
                      return <td key={y} className="pr-2">{dd(b.mid)} <span className="opacity-50">({dd(b.low)}–{dd(b.high)})</span></td>
                    })}
                    <td>{sgn(p.slope_decade, 2)}</td>
                    <td style={{ color: p.applied ? '#B42318' : undefined }}>{p.applied ? sigText(p.p) : `${sigText(p.p)}, not applied`}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      <Box
        title="Climate migration risk outlook: districts per level"
        note={`Risk score recomputed from the projected indicators, with the ${b0}–${b1} districts as the fixed yardstick and the same level cut-offs, so a general rise or fall shows up. Drought frequency and rainfall variability are held at their 2001–25 values.`}
        h="h-64"
      >
        <BarChart data={levelChart}>
          <CartesianGrid strokeDasharray="3 3" stroke="#0001" />
          <XAxis dataKey="year" tick={{ fontSize: 11 }} /><YAxis allowDecimals={false} tick={{ fontSize: 10 }} width={30} />
          <Tooltip /><Legend wrapperStyle={{ fontSize: 11 }} />
          {meta.levels.map((lv) => <Bar key={lv} dataKey={lv} stackId="a" fill={LEVEL_COLOR[lv]} />)}
        </BarChart>
      </Box>

      <div className={`${card} overflow-x-auto`}>
        <div className="mb-2 text-sm font-medium">District risk level by year (ranked by current risk)</div>
        <table className="w-full text-xs">
          <thead><tr className="text-left opacity-60"><th className="py-1 font-normal">District</th><th className="font-normal">Region</th><th className="font-normal">{b1}</th>{years.map((y) => <th key={y} className="font-normal">{y}</th>)}</tr></thead>
          <tbody>
            {outlook.map((x) => (
              <tr key={x.id} className="border-t border-black/5 hover:bg-black/[0.02]">
                <td className="py-1 pr-3 font-medium"><button onClick={() => { setSel(x.id); window.scrollTo({ top: 0, behavior: 'smooth' }) }} className="text-left underline-offset-2 hover:underline">{x.name}</button></td>
                <td className="pr-3 opacity-70">{x.region}</td>
                <td className="pr-2"><Pill level={x.risk.level} /></td>
                {years.map((y) => <td key={y} className="pr-2"><Pill level={x.future_risk[String(y)].level} /></td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

/* ── exports used by MaharashtraClimate.tsx ──────────────────── */

export const MigrationAnalysisTab = () => <Gate>{(d) => <MigrationView d={d} />}</Gate>
export const RiskScoreTab = () => <Gate>{(d) => <RiskView d={d} />}</Gate>
export const ProjectionTab = () => <Gate>{(d) => <ProjectionView d={d} />}</Gate>
