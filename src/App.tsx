import { useEffect, useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from 'recharts'

/* ── Types (match public/data/risk.json written by ml/train.py) ─ */

type Level = 'CRITICAL' | 'HIGH' | 'MODERATE' | 'LOW'

interface StateRisk {
  name: string; lat: number; lon: number
  prob: number; level: Level; delta: number
  events: number; affected: number; displaced_est: number
  dominant_hazard: string
  drivers: { label: string; z: number }[]
  trend: { m: string; p: number }[]
}
interface Metrics { roc_auc: number; pr_auc: number; brier?: number; recall_at_high?: number; precision_at_high?: number; base_rate?: number }
interface RiskData {
  meta: {
    generated: string; as_of: string; horizon_months: number; n_states: number
    train_years: number[]; val_years: number[]; test_years: number[]
    n_rows: number; positive_rate: number
    thresholds: { moderate: number; high: number; critical: number }
    model: string
    metrics: { gradient_boosting: Metrics; logistic_regression: Metrics; persistence: Metrics }
  }
  importance: { feature: string; label: string; importance: number }[]
  states: StateRisk[]
  alerts: { state: string; level: Level; prob: number; msg: string }[]
}

/* ── Palette ─────────────────────────────────────────────────── */

const RISK_COLOR: Record<string, string> = {
  CRITICAL: '#C0392B', HIGH: '#D4782A', MODERATE: '#3D7A5E', LOW: '#2E7D32',
}
const RISK_BG: Record<string, string> = {
  CRITICAL: 'rgba(192, 57, 43, 0.08)', HIGH: 'rgba(212, 120, 42, 0.08)',
  MODERATE: 'rgba(61, 122, 94, 0.08)', LOW: 'rgba(46, 125, 50, 0.08)',
}
const RISK_BORDER: Record<string, string> = {
  CRITICAL: 'rgba(192, 57, 43, 0.22)', HIGH: 'rgba(212, 120, 42, 0.22)',
  MODERATE: 'rgba(61, 122, 94, 0.22)', LOW: 'rgba(46, 125, 50, 0.22)',
}

const fmtNum = (n: number) =>
  n >= 1e7 ? `${(n / 1e7).toFixed(1)} Cr` : n >= 1e5 ? `${(n / 1e5).toFixed(1)} L` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}k` : `${Math.round(n)}`

/* ── Data hook ───────────────────────────────────────────────── */

function useRiskData() {
  const [data, setData] = useState<RiskData | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/risk.json`)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then(setData)
      .catch(e => setError(String(e)))
  }, [])
  return { data, error }
}

/* ── Sub-components ──────────────────────────────────────────── */

function RiskBadge({ level }: { level: string }) {
  return (
    <span
      style={{ color: RISK_COLOR[level], backgroundColor: RISK_BG[level], borderColor: RISK_BORDER[level] }}
      className="inline-flex items-center gap-1.5 px-2.5 py-0.5 text-[10px] font-mono font-semibold tracking-widest border rounded-sm select-none"
    >
      {level === 'CRITICAL' && <span className="w-1.5 h-1.5 rounded-full bg-[#C0392B] risk-pulse-red inline-block flex-shrink-0" />}
      {level === 'HIGH' && <span className="w-1.5 h-1.5 rounded-full bg-[#D4782A] risk-pulse-amber inline-block flex-shrink-0" />}
      {level}
    </span>
  )
}

function ScoreBar({ level, value }: { level: string; value: number }) {
  return (
    <div className="flex items-center gap-3">
      <div className="flex-1 h-0.5 bg-black/[0.06] rounded-full overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${Math.min(100, value)}%`, backgroundColor: RISK_COLOR[level] }} />
      </div>
      <span style={{ fontFamily: 'JetBrains Mono, monospace', color: RISK_COLOR[level] }} className="text-sm font-semibold w-12 text-right">
        {value.toFixed(0)}%
      </span>
    </div>
  )
}

function Cell({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-black/30 text-[9px] font-mono tracking-widest uppercase mb-0.5">{label}</p>
      <p style={{ fontFamily: 'JetBrains Mono, monospace' }} className="text-xs text-black/60 capitalize">{value}</p>
    </div>
  )
}

function Notice({ title, body }: { title: string; body: string }) {
  return (
    <div className="min-h-screen flex items-center justify-center bg-[#F4F3EC] px-6">
      <div className="max-w-lg text-center">
        <h1 style={{ fontFamily: 'Fraunces, serif' }} className="text-2xl font-semibold mb-3">{title}</h1>
        <p className="text-black/45 text-sm leading-relaxed whitespace-pre-line">{body}</p>
      </div>
    </div>
  )
}

/* ── Main ────────────────────────────────────────────────────── */

export default function App() {
  const { data, error } = useRiskData()
  const [filter, setFilter] = useState('ALL')
  const [dismissed, setDismissed] = useState<string[]>([])
  const [navOpen, setNavOpen] = useState(false)

  if (error) {
    return (
      <Notice
        title="Risk data not found"
        body={`Could not load public/data/risk.json (${error}).\nRun the pipeline first:\npython ml/fetch_climate.py → convert_emdat.py → prepare_data.py → train.py`}
      />
    )
  }
  if (!data) return <Notice title="Loading…" body="Reading model output" />

  const { meta, states, alerts, importance } = data
  const th = meta.thresholds
  const gb = meta.metrics.gradient_boosting
  const lr = meta.metrics.logistic_regression
  const pers = meta.metrics.persistence
  const horizon = meta.horizon_months

  const displayed = filter === 'ALL' ? states : states.filter(r => r.level === filter)
  const activeAlerts = alerts.filter(a => !dismissed.includes(a.state))
  const nCritical = states.filter(s => s.level === 'CRITICAL').length
  const nHigh = states.filter(s => s.level === 'HIGH').length

  const tickerItems = alerts.slice(0, 6)
  const doubledTicker = [...tickerItems, ...tickerItems]

  // top-3 states trend chart
  const top3 = states.slice(0, 3)
  const chartData = top3[0].trend.map((pt, i) => ({
    m: pt.m,
    a: top3[0].trend[i]?.p, b: top3[1].trend[i]?.p, c: top3[2].trend[i]?.p,
  }))
  const lineColors = ['#C0392B', '#D4782A', '#4A7C59']
  const maxImp = Math.max(...importance.map(i => i.importance), 1e-9)

  const levelDefs: { level: Level; range: string; desc: string }[] = [
    { level: 'CRITICAL', range: `≥ ${(th.critical * 100).toFixed(0)}%`, desc: `Top ~7% of risk levels seen in validation years. High chance of a high-impact disaster that displaces people within ${horizon} months. Pre-position relief and shelters.` },
    { level: 'HIGH', range: `${(th.high * 100).toFixed(0)}–${(th.critical * 100).toFixed(0)}%`, desc: 'Risk clearly above normal. Contingency plans should be reviewed and district teams alerted.' },
    { level: 'MODERATE', range: `${(th.moderate * 100).toFixed(0)}–${(th.high * 100).toFixed(0)}%`, desc: 'Somewhat elevated conditions. Preparedness checks and closer monitoring recommended.' },
    { level: 'LOW', range: `< ${(th.moderate * 100).toFixed(0)}%`, desc: 'Below-average risk for this period. Routine monitoring continues.' },
  ]

  const methodSteps = [
    { n: '01', title: 'Data Collection', desc: 'Monthly temperature and rainfall for every Indian state come from NASA POWER (1990 onwards). Disaster records (floods, cyclones, droughts, heat waves) come from the EM-DAT database.' },
    { n: '02', title: 'Feature Engineering', desc: 'Rainfall and temperature are converted into anomalies against the 1991–2020 normal. Past-disaster counts, season and state characteristics (coastal, Himalayan) are added.' },
    { n: '03', title: 'Model Training', desc: `A gradient-boosting classifier is trained on ${meta.train_years[0]}–${meta.train_years[1]}, calibrated on ${meta.val_years[0]}–${meta.val_years[1]} and tested on ${meta.test_years[0]}–${meta.test_years[1]}. Splits are by time, never random.` },
    { n: '04', title: 'Risk Scoring', desc: `Each state gets a calibrated probability of a high-impact disaster (one likely to displace people) in the next ${horizon} months, mapped to LOW / MODERATE / HIGH / CRITICAL.` },
  ]

  const modelRows = [
    { name: 'Gradient boosting (ours)', m: gb, bold: true },
    { name: 'Logistic regression', m: lr, bold: false },
    { name: 'Persistence (recent disasters)', m: pers, bold: false },
  ]

  return (
    <div className="min-h-screen bg-[#F4F3EC]" style={{ color: '#1A2418' }}>

      {/* ── Navigation ───────────────────────────────────────── */}
      <header className="fixed top-0 left-0 right-0 z-50 bg-[#F4F3EC]/90 backdrop-blur-md border-b border-black/[0.07]">
        <div className="max-w-7xl mx-auto px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-7 h-7 rounded-sm bg-[#4A7C59] flex items-center justify-center">
              <span className="text-white text-[10px] font-bold font-mono">CW</span>
            </div>
            <span style={{ fontFamily: 'Fraunces, serif' }} className="font-semibold text-base tracking-tight">
              ClimateWatch <span className="text-black/25 font-light">/ India EWS</span>
            </span>
          </div>

          <nav className="hidden md:flex items-center gap-8">
            {['Dashboard', 'Risk Map', 'Methodology', 'Alerts', 'Model'].map(item => (
              <a key={item} href={`#${item.toLowerCase().replace(' ', '-')}`}
                className="text-black/35 hover:text-black/70 text-xs tracking-widest uppercase transition-colors font-medium">
                {item}
              </a>
            ))}
          </nav>

          <div className="flex items-center gap-3">
            <div className="hidden md:flex items-center gap-2 text-xs font-mono">
              <span className="w-1.5 h-1.5 rounded-full bg-[#4A7C59]" />
              <span className="text-black/30">DATA AS OF {meta.as_of}</span>
            </div>
            <button onClick={() => setNavOpen(!navOpen)} className="md:hidden text-black/40 hover:text-black p-1 font-mono text-lg">
              {navOpen ? '✕' : '☰'}
            </button>
          </div>
        </div>

        {navOpen && (
          <div className="md:hidden bg-[#EDECE4] border-t border-black/[0.06] px-6 py-4 flex flex-col gap-4">
            {['Dashboard', 'Risk Map', 'Methodology', 'Alerts', 'Model'].map(item => (
              <a key={item} href={`#${item.toLowerCase().replace(' ', '-')}`} onClick={() => setNavOpen(false)}
                className="text-black/40 hover:text-black/80 text-sm tracking-widest uppercase transition-colors">
                {item}
              </a>
            ))}
          </div>
        )}
      </header>

      {/* ── Hero ─────────────────────────────────────────────── */}
      <section id="dashboard" className="relative pt-14 min-h-screen flex flex-col justify-center overflow-hidden">
        <div className="absolute inset-0 pointer-events-none" style={{
          backgroundImage: 'linear-gradient(rgba(74,124,89,0.045) 1px, transparent 1px), linear-gradient(90deg, rgba(74,124,89,0.045) 1px, transparent 1px)',
          backgroundSize: '48px 48px',
        }} />
        <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[500px] rounded-full pointer-events-none"
          style={{ background: 'radial-gradient(ellipse, rgba(184,216,192,0.35) 0%, transparent 70%)' }} />

        <div className="max-w-7xl mx-auto px-6 py-24 relative z-10">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            <div>
              <div className="inline-flex items-center gap-2.5 border border-[#C0392B]/25 bg-[#C0392B]/5 rounded-sm px-3 py-1.5 mb-8">
                <span className="w-1.5 h-1.5 rounded-full bg-[#C0392B] risk-pulse-red" />
                <span className="text-[#C0392B] text-[10px] font-mono font-semibold tracking-[0.15em] uppercase">
                  {nCritical} Critical · {nHigh} High-risk states
                </span>
              </div>

              <h1 style={{ fontFamily: 'Fraunces, serif' }}
                className="text-5xl lg:text-6xl font-semibold leading-[1.1] tracking-tight mb-6 text-[#1A2418]">
                Early Warning<br />for Climate-<br />
                <em className="not-italic text-[#4A7C59]">Induced Migration</em>
              </h1>

              <p className="text-black/45 text-lg leading-relaxed mb-10 max-w-lg">
                Predicting which Indian states are most likely to face disasters severe enough to
                displace people. A machine-learning model trained on historical climate and disaster
                records issues a {horizon}-month risk forecast for every state.
              </p>

              <div className="flex flex-wrap gap-4">
                <a href="#risk-map" className="bg-[#4A7C59] hover:bg-[#3D6B4E] text-white text-sm font-semibold px-6 py-3 rounded-sm transition-colors tracking-wide">
                  View State Risk →
                </a>
                <a href="#methodology" className="border border-black/15 hover:border-black/30 text-black/45 hover:text-black/80 text-sm font-medium px-6 py-3 rounded-sm transition-all tracking-wide">
                  How It Works
                </a>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              {[
                { label: 'States Monitored', value: `${meta.n_states}`, sub: 'India', color: '#3D7A5E' },
                { label: 'High / Critical', value: `${nCritical + nHigh}`, sub: `next ${horizon} months`, color: '#C0392B' },
                { label: 'Critical Alerts', value: `${nCritical}`, sub: `≥ ${(th.critical * 100).toFixed(0)}% probability`, color: '#C0392B' },
                { label: 'Model ROC-AUC', value: gb.roc_auc.toFixed(2), sub: `test ${meta.test_years[0]}–${meta.test_years[1]}`, color: '#2E7D32' },
                { label: 'Training Data', value: `${meta.train_years[0]}–${meta.test_years[1]}`, sub: `${meta.n_rows.toLocaleString()} state-months`, color: '#3D7A5E' },
                { label: 'Data As Of', value: meta.as_of, sub: `built ${meta.generated}`, color: '#D4782A' },
              ].map(stat => (
                <div key={stat.label} className="bg-white/70 border border-black/[0.07] rounded p-4 hover:border-[#4A7C59]/25 transition-colors backdrop-blur-sm">
                  <p className="text-black/30 text-[10px] font-mono tracking-widest uppercase mb-2">{stat.label}</p>
                  <p style={{ fontFamily: 'Fraunces, serif', color: stat.color }} className="text-3xl font-semibold mb-0.5">{stat.value}</p>
                  <p className="text-black/30 text-xs font-mono">{stat.sub}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="mt-16 flex justify-center">
            <div className="flex flex-col items-center gap-2 text-black/15">
              <div className="w-px h-8 bg-gradient-to-b from-transparent to-black/15" />
              <span className="text-[10px] font-mono tracking-widest">SCROLL</span>
            </div>
          </div>
        </div>
      </section>

      {/* ── Alert Ticker ─────────────────────────────────────── */}
      {tickerItems.length > 0 && (
        <div className="bg-[#EAE9E1] border-y border-black/[0.07] py-3 ticker-wrap" id="alerts">
          <div className="ticker-content">
            {doubledTicker.map((a, i) => (
              <div key={i} className="flex items-center gap-4 pr-12 flex-shrink-0">
                <RiskBadge level={a.level} />
                <span className="text-black/55 text-xs font-medium">{a.state}</span>
                <span className="text-black/35 text-xs">{a.prob.toFixed(0)}% risk in next {horizon} months</span>
                <span className="text-black/15 text-xs">·</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Active Alerts ────────────────────────────────────── */}
      <section className="max-w-7xl mx-auto px-6 py-16">
        <div className="flex items-center justify-between mb-8">
          <div>
            <p className="text-[10px] font-mono text-black/25 tracking-widest uppercase mb-1">Latest Forecast</p>
            <h2 style={{ fontFamily: 'Fraunces, serif' }} className="text-2xl font-semibold">Active Alerts</h2>
          </div>
          <div className="text-xs font-mono text-black/30">As of {meta.as_of}</div>
        </div>

        <div className="space-y-2">
          {activeAlerts.map(alert => (
            <div key={alert.state}
              className="flex items-start gap-4 bg-white/60 border border-black/[0.07] rounded p-4 hover:border-black/10 transition-colors group"
              style={{ borderLeft: `2px solid ${RISK_COLOR[alert.level]}` }}>
              <RiskBadge level={alert.level} />
              <div className="flex-1 min-w-0">
                <span className="text-black/50 text-xs font-mono mr-2">{alert.state}</span>
                <span className="text-black/60 text-sm">{alert.msg}</span>
              </div>
              <button onClick={() => setDismissed(prev => [...prev, alert.state])}
                className="text-black/15 hover:text-black/50 transition-colors opacity-0 group-hover:opacity-100 text-xs font-mono">
                ✕
              </button>
            </div>
          ))}
          {activeAlerts.length === 0 && (
            <div className="text-center py-8 text-black/25 text-sm font-mono">No active alerts</div>
          )}
        </div>
      </section>

      {/* ── State Risk Cards ─────────────────────────────────── */}
      <section id="risk-map" className="max-w-7xl mx-auto px-6 pb-20">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-8">
          <div>
            <p className="text-[10px] font-mono text-black/25 tracking-widest uppercase mb-1">Vulnerability Assessment</p>
            <h2 style={{ fontFamily: 'Fraunces, serif' }} className="text-2xl font-semibold">State-wise Risk</h2>
          </div>
          <div className="flex items-center gap-1 bg-white/60 border border-black/[0.07] rounded p-1">
            {['ALL', 'CRITICAL', 'HIGH', 'MODERATE', 'LOW'].map(f => (
              <button key={f} onClick={() => setFilter(f)}
                className={`text-[10px] font-mono tracking-widest px-3 py-1.5 rounded-sm transition-all ${
                  filter === f ? 'bg-[#4A7C59] text-white' : 'text-black/30 hover:text-black/60'}`}>
                {f}
              </button>
            ))}
          </div>
        </div>

        <div className="grid lg:grid-cols-2 gap-3">
          {displayed.map(r => (
            <div key={r.name}
              className="bg-white/60 border border-black/[0.07] rounded p-5 hover:border-[#4A7C59]/25 transition-all cursor-default backdrop-blur-sm"
              style={{ borderLeft: `2px solid ${RISK_COLOR[r.level]}` }}>
              <div className="flex items-start justify-between mb-4">
                <div className="flex-1 min-w-0 pr-4">
                  <h3 className="font-semibold text-sm mb-0.5 text-[#1A2418]/80">{r.name}</h3>
                  <p className="text-black/30 text-xs font-mono">
                    {r.drivers.length ? r.drivers.map(d => d.label).join(' · ') : 'No elevated signals'}
                  </p>
                </div>
                <RiskBadge level={r.level} />
              </div>

              <ScoreBar level={r.level} value={r.prob} />

              <div className="grid grid-cols-4 gap-3 mt-4">
                <Cell label="Past events" value={`${r.events}`} />
                <Cell label="Affected (hist.)" value={fmtNum(r.affected)} />
                <Cell label="Δ vs last month" value={`${r.delta > 0 ? '+' : ''}${r.delta.toFixed(1)} pt`} />
                <Cell label="Main hazard" value={r.dominant_hazard} />
              </div>

              <div className="flex items-center justify-between mt-4 pt-4 border-t border-black/[0.05]">
                <span className="text-black/25 text-[10px] font-mono">{r.lat.toFixed(1)}°N · {r.lon.toFixed(1)}°E</span>
                <span className="text-black/25 text-[10px] font-mono">Forecast as of {meta.as_of}</span>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Risk Trend Chart ─────────────────────────────────── */}
      <section className="max-w-7xl mx-auto px-6 pb-20">
        <div className="bg-white/60 border border-black/[0.07] rounded p-8 backdrop-blur-sm">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-8">
            <div>
              <p className="text-[10px] font-mono text-black/25 tracking-widest uppercase mb-1">Last 10 months</p>
              <h2 style={{ fontFamily: 'Fraunces, serif' }} className="text-xl font-semibold">Risk Trajectory</h2>
              <p className="text-black/35 text-sm mt-1">Predicted probability (%) of a high-impact disaster, top 3 states</p>
            </div>
            <div className="flex gap-5 text-xs font-mono">
              {top3.map((s, i) => (
                <div key={s.name} className="flex items-center gap-1.5">
                  <div className="w-5 h-0.5 rounded" style={{ backgroundColor: lineColors[i] }} />
                  <span className="text-black/45">{s.name}</span>
                </div>
              ))}
            </div>
          </div>

          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={chartData} margin={{ top: 5, right: 5, left: -20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="2 6" stroke="rgba(0,0,0,0.05)" />
              <XAxis dataKey="m" tick={{ fill: 'rgba(0,0,0,0.35)', fontSize: 11, fontFamily: 'JetBrains Mono' }} axisLine={false} tickLine={false} />
              <YAxis domain={[0, 100]} tick={{ fill: 'rgba(0,0,0,0.35)', fontSize: 11, fontFamily: 'JetBrains Mono' }} axisLine={false} tickLine={false} />
              <Tooltip />
              <ReferenceLine y={th.high * 100} stroke="rgba(212,120,42,0.40)" strokeDasharray="4 4"
                label={{ value: 'HIGH', fill: 'rgba(212,120,42,0.7)', fontSize: 10, fontFamily: 'JetBrains Mono' }} />
              <ReferenceLine y={th.critical * 100} stroke="rgba(192,57,43,0.40)" strokeDasharray="4 4"
                label={{ value: 'CRITICAL', fill: 'rgba(192,57,43,0.7)', fontSize: 10, fontFamily: 'JetBrains Mono' }} />
              {top3.map((s, i) => (
                <Line key={s.name} type="monotone" dataKey={['a', 'b', 'c'][i]} stroke={lineColors[i]}
                  strokeWidth={2} dot={false} name={s.name} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* ── Methodology ──────────────────────────────────────── */}
      <section id="methodology" className="max-w-7xl mx-auto px-6 pb-20">
        <div className="grid lg:grid-cols-[280px_1fr] gap-16">
          <div>
            <p className="text-[10px] font-mono text-black/25 tracking-widest uppercase mb-3">How It Works</p>
            <h2 style={{ fontFamily: 'Fraunces, serif' }} className="text-3xl font-semibold leading-snug mb-4">The Prediction Pipeline</h2>
            <p className="text-black/40 text-sm leading-relaxed">
              Four stages turn historical climate and disaster data into a state-level displacement-risk forecast.
            </p>
          </div>
          <div className="grid sm:grid-cols-2 gap-4">
            {methodSteps.map(step => (
              <div key={step.n} className="bg-white/60 border border-black/[0.07] rounded p-6 hover:border-[#4A7C59]/25 transition-colors">
                <div className="flex items-start gap-4 mb-3">
                  <span style={{ fontFamily: 'JetBrains Mono, monospace' }} className="text-[10px] font-semibold text-black/25 tracking-widest mt-0.5">{step.n}</span>
                  <h3 className="font-semibold text-sm text-[#1A2418]">{step.title}</h3>
                </div>
                <p className="text-black/45 text-xs leading-relaxed pl-8">{step.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── What drives the model ────────────────────────────── */}
      <section className="border-t border-black/[0.05] max-w-7xl mx-auto px-6 py-20">
        <div className="text-center mb-12">
          <p className="text-[10px] font-mono text-black/25 tracking-widest uppercase mb-3">Model Drivers</p>
          <h2 style={{ fontFamily: 'Fraunces, serif' }} className="text-3xl font-semibold">What the Model Relies On</h2>
          <p className="text-black/40 text-sm mt-3">Permutation importance on the test years (share of total)</p>
        </div>
        <div className="max-w-2xl mx-auto space-y-3">
          {importance.map(f => (
            <div key={f.feature} className="flex items-center gap-4">
              <span className="w-56 text-xs text-black/55 truncate">{f.label}</span>
              <div className="flex-1 h-2 bg-black/[0.06] rounded-full overflow-hidden">
                <div className="h-full bg-[#4A7C59] rounded-full" style={{ width: `${(f.importance / maxImp) * 100}%` }} />
              </div>
              <span style={{ fontFamily: 'JetBrains Mono, monospace' }} className="w-12 text-right text-xs text-black/45">
                {(f.importance * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
      </section>

      {/* ── Alert Level Guide ────────────────────────────────── */}
      <section className="max-w-7xl mx-auto px-6 pb-20">
        <div className="bg-white/60 border border-black/[0.07] rounded p-8 lg:p-10 backdrop-blur-sm">
          <div className="mb-8">
            <p className="text-[10px] font-mono text-black/25 tracking-widest uppercase mb-2">Reference</p>
            <h2 style={{ fontFamily: 'Fraunces, serif' }} className="text-2xl font-semibold">Alert Level Definitions</h2>
          </div>
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {levelDefs.map(item => (
              <div key={item.level} className="rounded p-5"
                style={{ backgroundColor: RISK_BG[item.level], border: `1px solid ${RISK_BORDER[item.level]}` }}>
                <RiskBadge level={item.level} />
                <p style={{ fontFamily: 'JetBrains Mono, monospace', color: RISK_COLOR[item.level] }} className="text-2xl font-semibold mt-3 mb-2">
                  {item.range}
                </p>
                <p className="text-black/45 text-xs leading-relaxed">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Model Report ─────────────────────────────────────── */}
      <section id="model" className="border-t border-black/[0.05] max-w-7xl mx-auto px-6 py-20">
        <div className="grid lg:grid-cols-2 gap-12 items-start">
          <div>
            <p className="text-[10px] font-mono text-black/25 tracking-widest uppercase mb-3">Evaluation</p>
            <h2 style={{ fontFamily: 'Fraunces, serif' }} className="text-4xl font-semibold leading-tight mb-4">
              How Good Is<br />the Forecast?
            </h2>
            <p className="text-black/45 text-base leading-relaxed mb-6">
              Tested on {meta.test_years[0]}–{meta.test_years[1]}, years the model never saw during training.
              About {(gb.base_rate ?? meta.positive_rate) * 100 > 0 ? ((gb.base_rate ?? meta.positive_rate) * 100).toFixed(1) : '0'}% of
              state-months were followed by a high-impact disaster, so a useful model must beat that base rate.
            </p>
            <div className="bg-white/60 border border-black/[0.07] rounded p-5 text-xs text-black/50 leading-relaxed">
              <p className="font-semibold text-black/70 mb-2">Limitations</p>
              <ul className="space-y-1.5 list-disc pl-4">
                <li>No official month-wise migration count exists for Indian states. Displacement is estimated from disaster records (homeless + a share of affected people).</li>
                <li>Climate is taken at one point per state, so local extremes can be missed.</li>
                <li>Forecasts show risk, not certainty. Use them to prioritise preparedness, not as a prediction of what will happen.</li>
              </ul>
            </div>
          </div>

          <div className="bg-white/60 border border-black/[0.07] rounded overflow-hidden">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-black/30 font-mono tracking-widest uppercase text-[10px] border-b border-black/[0.06]">
                  <th className="text-left p-4">Model</th>
                  <th className="text-right p-4">ROC-AUC</th>
                  <th className="text-right p-4">PR-AUC</th>
                </tr>
              </thead>
              <tbody>
                {modelRows.map(r => (
                  <tr key={r.name} className="border-b border-black/[0.04] last:border-0">
                    <td className={`p-4 ${r.bold ? 'font-semibold text-[#1A2418]' : 'text-black/55'}`}>{r.name}</td>
                    <td style={{ fontFamily: 'JetBrains Mono, monospace' }} className="p-4 text-right">{r.m.roc_auc.toFixed(3)}</td>
                    <td style={{ fontFamily: 'JetBrains Mono, monospace' }} className="p-4 text-right">{r.m.pr_auc.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="grid grid-cols-3 gap-3 p-4 bg-black/[0.02] border-t border-black/[0.06]">
              <Cell label="Recall @ HIGH" value={gb.recall_at_high !== undefined ? `${(gb.recall_at_high * 100).toFixed(0)}%` : '—'} />
              <Cell label="Precision @ HIGH" value={gb.precision_at_high !== undefined ? `${(gb.precision_at_high * 100).toFixed(0)}%` : '—'} />
              <Cell label="Brier score" value={gb.brier !== undefined ? gb.brier.toFixed(3) : '—'} />
            </div>
          </div>
        </div>
      </section>

      {/* ── Footer ───────────────────────────────────────────── */}
      <footer className="border-t border-black/[0.06] bg-[#ECEAE2]">
        <div className="max-w-7xl mx-auto px-6 py-12">
          <div className="grid sm:grid-cols-2 gap-8 mb-10">
            <div>
              <div className="flex items-center gap-2.5 mb-4">
                <div className="w-6 h-6 rounded-sm bg-[#4A7C59] flex items-center justify-center">
                  <span className="text-white text-[9px] font-bold font-mono">CW</span>
                </div>
                <span style={{ fontFamily: 'Fraunces, serif' }} className="text-sm font-semibold">ClimateWatch India EWS</span>
              </div>
              <p className="text-black/35 text-xs leading-relaxed max-w-sm">
                Early warning for climate-induced migration in India. Academic project built with
                Python (scikit-learn) and React.
              </p>
            </div>
            <div>
              <h4 className="text-black/30 text-[10px] font-mono tracking-widest uppercase mb-4">Data Sources</h4>
              <ul className="space-y-2 text-xs text-black/40">
                <li>NASA POWER — monthly temperature and rainfall</li>
                <li>EM-DAT — disaster records for India</li>
                <li>Census of India — migration context (optional)</li>
              </ul>
            </div>
          </div>
          <div className="pt-8 border-t border-black/[0.06] text-black/30 text-xs font-mono">
            © 2026 ClimateWatch India EWS · Forecast as of {meta.as_of} · Model: {meta.model}
          </div>
        </div>
      </footer>
    </div>
  )
}
