import { useState } from 'react'
import type { Reliability, StoryData } from '../types'
import { DAY, duration, formatDate, linear, logScale, project, time } from '../lib/math'
import { Evidence, SectionHead, Segment, SourceLine } from './ui'
import { useCompactChart } from '../lib/useCompactChart'
import CostFuture from './CostFuture'

const PACES = [
  { value: '0', label: 'No further progress', short: 'Flat', factor: 0 },
  { value: '0.5', label: 'Half the historical pace', short: 'Half pace', factor: 0.5 },
  { value: '1', label: 'Historical pace continues', short: 'Same pace', factor: 1 },
] as const

export default function Future({ data }: { data: StoryData }) {
  const mobile = useCompactChart()
  const [reliability, setReliability] = useState<Reliability>('p50')
  const [pace, setPace] = useState('1')
  const [months, setMonths] = useState(12)
  const trend = data.trends[reliability]
  const path = PACES.find(p => p.value === pace)!
  const days = months * 365.25 / 12
  const value = project(trend.anchor_minutes, days, trend.doubling_days, path.factor)
  const target = new Date(time(trend.anchor_date) + days * DAY).toISOString().slice(0, 10)
  const W = mobile ? 600 : 1040, H = 390, left = 65
  const right = mobile ? 100 : 140, top = 28, bottom = H - 56
  const x = linear([0, 24], [left, W - right])
  const maximum = project(trend.anchor_minutes, 2 * 365.25, trend.doubling_days, 1) * 1.3
  const y = logScale([Math.min(60, trend.anchor_minutes / 2), maximum], [bottom, top])
  const ticks = [60, 240, 960, 1440, 10080, 43200, 129600].filter(t =>
    t < maximum && t >= Math.min(60, trend.anchor_minutes / 2))
  const selectedModel = data.horizons.find(m => m.id === trend.anchor_id)!
  const crossesLimit = value > data.benchmark.reliable_range_minutes
  const reliableRange = data.benchmark.reliable_range_minutes
  const skill = trend.backtest.skill_ratio

  return <section id="future" className="chapter future-chapter">
    <SectionHead number="04" label="The possible futures" title={<>The next chapter<br />is not a straight line.</>}>
      We can extend a measured trend. We cannot know that it will continue.
      Change the assumption and see how different the future becomes.
    </SectionHead>
    <div className="scenario-banner"><span className="scenario-symbol" aria-hidden="true">◇</span>
      Conditional scenario <span>Not a prediction. Not a GPT release schedule.</span></div>
    <div className="chart-toolbar"><span className="chart-label">Human task duration</span>
      <Segment value={reliability} onChange={setReliability} label="Scenario success rate" options={[
        { value: 'p50', label: '50% success' }, { value: 'p80', label: '80% success' },
      ]} />
    </div>
    <div className="future-readout" aria-live="polite">
      <div><span className="stat-label">If {path.factor === 0 ? 'progress stops here' : path.factor === 0.5
        ? 'progress slows to half pace' : 'the historical pace continues'}</span>
        <strong>{duration(value)}</strong>
        <span className="future-date">by {formatDate(target, true)}</span></div>
      <div className="future-explainer"><span className="mini-rule" />
        <p>{reliability === 'p50' ? '50' : '80'}% success on comparable benchmark tasks.
          {crossesLimit ? ' This extends beyond what the current task suite can measure reliably.'
            : ' This is a calculated path, not a new measurement.'}</p></div>
    </div>
    <svg className="data-chart future-chart" viewBox={`0 0 ${W} ${H}`} role="img"
      aria-label={`Three conditional paths over 24 months. Selected: ${path.label},
        ${duration(value)} at ${months} months after ${trend.anchor_date}.`}>
      {ticks.map(tick => <g key={tick} className="grid-line">
        <line x1={left} x2={W - right + 80} y1={y(tick)} y2={y(tick)} />
        <text x={left - 14} y={y(tick) + 4} textAnchor="end">{duration(tick, true)}</text>
      </g>)}
      {[0, 6, 12, 18, 24].map(month => <text key={month} className="axis-caption"
        x={x(month)} y={H - 27} textAnchor="middle">{month === 0 ? 'Measured' : `+${month}mo`}</text>)}
      {maximum > reliableRange && <g className="chart-limit">
        <line x1={left} x2={W - right + 80} y1={y(reliableRange)} y2={y(reliableRange)} />
        <text x={left + 10} y={y(reliableRange) - 10}>{duration(reliableRange, true)} · current measurement limit</text>
      </g>}
      {PACES.map(p => {
        const selected = pace === p.value
        const d = Array.from({ length: 49 }, (_, i) => {
          const m = i / 2
          return `${i ? 'L' : 'M'}${x(m)},${y(project(trend.anchor_minutes, m * 365.25 / 12, trend.doubling_days, p.factor))}`
        }).join(' ')
        return <g key={p.value} className={selected ? 'scenario-path active' : 'scenario-path'}>
          <path d={d} />
          <text x={x(24) + 14} y={y(project(trend.anchor_minutes, 2 * 365.25, trend.doubling_days, p.factor)) + 4}>
            {p.short}</text>
        </g>
      })}
      <line className="scrub-line" x1={x(months)} x2={x(months)} y1={top} y2={bottom} />
      <circle className="future-point" cx={x(months)} cy={y(value)} r="6" />
      <circle className="anchor-point" cx={left} cy={y(trend.anchor_minutes)} r="5" />
      <text className="axis-caption" x={left} y={H - 3}>
        {mobile ? 'Months after baseline · log scale' : 'Months after the measured baseline · logarithmic scale'}
      </text>
    </svg>
    <div className="scenario-controls">
      <label className="select-label">Assumption<select value={pace} onChange={e => setPace(e.target.value)}>
        {PACES.map(p => <option key={p.value} value={p.value}>{p.label}</option>)}
      </select></label>
      <div className="horizon-slider"><label htmlFor="future-months">Time after baseline <strong>{months} months</strong></label>
        <input id="future-months" type="range" min="0" max="24" step="1" value={months}
          onChange={e => setMonths(Number(e.target.value))} aria-valuetext={`${months} months after the measured baseline`} />
      </div>
      <button className="text-button" onClick={() => { setMonths(12); setPace('1'); setReliability('p50') }}>Reset ↺</button>
    </div>
    <div className="scenario-baseline">
      <div><span className="stat-label">Measured starting point</span><strong>{duration(trend.anchor_minutes)}</strong>
        <span>{selectedModel.name} · {formatDate(trend.anchor_date, true)}</span></div>
      <div><span className="stat-label">Fitted historical doubling time</span>
        <strong>{(trend.doubling_days / (365.25 / 12)).toFixed(1)} months</strong>
        <span>{trend.n} selected observations · {trend.start.slice(0, 4)} to {trend.end.slice(0, 4)}</span></div>
      <div><span className="stat-label">What the paths mean</span>
        <p>Flat, half pace, and the fitted pace. These are assumptions, not confidence bounds.</p></div>
    </div>
    <SourceLine href="https://metr.org/time-horizons/" label="Derived from METR TH 1.1">
      No probability is assigned to any path · progress may slow, stop, or change direction
    </SourceLine>
    <Evidence title="Show the assumptions and the historical check">
      <p>We fit a straight line to log₂ of the measured task horizon against release date, using the
        highest eligible observation on each date. Points above 16 hours are excluded from fitting.
        The starting point is the highest eligible measurement, at its actual release date,
        not a fabricated measurement today. This differs from METR’s own headline trend selection.</p>
      <p>The scenario is baseline × 2^(days × pace ÷ doubling time). Flat uses pace 0; half pace uses
        0.5; same pace uses 1. They are deliberately chosen sensitivity cases, not estimated
        probabilities. Results beyond the measurement limit are extrapolations outside supported
        task lengths. A “day” is 24 hours of equivalent human task time, not a working day.</p>
      <p>A robust alternative gives a doubling time of {(trend.robust_doubling_days / (365.25 / 12)).toFixed(1)}
        {' '}months. That difference is another reason not to read one curve as the future.</p>
      {trend.backtest.mae_log2 !== null && trend.backtest.baseline_mae_log2 !== null ? <p>
        In a retrospective next-release check over {trend.backtest.n} observations, the trend’s
        mean absolute log₂ error was {trend.backtest.mae_log2.toFixed(2)}, versus
        {' '}{trend.backtest.baseline_mae_log2.toFixed(2)} for keeping the last value.
        {' '}{skill !== null && skill < 1 ? 'The trend did better on that check.' : 'The trend did not beat the simple baseline.'}
        {' '}This uses today’s revised evaluation data, not snapshots known at the time.
        It does not validate a two-year forecast. We therefore present scenarios, not predictions.</p>
        : <p>There are too few distinct earlier release dates to run the historical check.
          This path is an unvalidated conditional scenario.</p>}
    </Evidence>
    <CostFuture data={data} />
  </section>
}
