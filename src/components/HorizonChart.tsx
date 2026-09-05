import { useId } from 'react'
import { chartDateWindow, duration, linear, logScale, time } from '../lib/math'
import type { Horizon, Reliability } from '../types'
import { useCompactChart } from '../lib/useCompactChart'

export default function HorizonChart({ models, reliability, selected, onSelect, reliableRange, compact = false }: {
  models: Horizon[]; reliability: Reliability; selected: string
  onSelect: (id: string) => void; reliableRange: number; compact?: boolean
}) {
  const title = useId()
  const mobile = useCompactChart()
  const current = models.find(m => m.id === selected) ?? models[models.length - 1]
  const W = mobile ? 600 : 1040, H = compact ? 250 : 430
  const left = 64, right = 24, top = 30, bottom = H - 48
  const maximum = Math.max(1440, ...models.map(m => m[reliability].estimate * 1.15),
    compact ? 0 : current[reliability].ci_high * 1.1)
  const window = chartDateWindow(models.map(model => model.release_date))
  const x = linear(window.domain, [left, W - right])
  const y = logScale([0.5, maximum], [bottom, top])
  const ticks = [1, 5, 15, 60, 240, 960, 2880].filter(t => t < maximum)
  const visible = models
  const leaders: Horizon[] = []
  let best = 0
  for (const model of visible) {
    if (model[reliability].estimate > best) {
      best = model[reliability].estimate
      leaders.push(model)
    }
  }
  const path = leaders.map((m, i) =>
    `${i ? 'L' : 'M'}${x(time(m.release_date))},${y(m[reliability].estimate)}`).join(' ')
  function step(direction: number) {
    const index = visible.findIndex(m => m.id === current.id)
    onSelect(visible[(index + direction + visible.length) % visible.length].id)
  }

  return <svg className={`data-chart horizon-chart ${compact ? 'compact-chart' : ''}`}
    viewBox={`0 0 ${W} ${H}`} role="group" aria-labelledby={title}
    onKeyDown={event => {
      if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
        event.preventDefault(); step(event.key === 'ArrowRight' ? 1 : -1)
      }
    }}>
    <title id={title}>Human task duration at {reliability === 'p50' ? '50' : '80'}% success.
      Select a model or use left and right arrow keys. The full values are in the measurement table.</title>
    {ticks.map(tick => <g key={tick} className="grid-line">
      <line x1={left} x2={W - right} y1={y(tick)} y2={y(tick)} />
      <text x={left - 14} y={y(tick) + 4} textAnchor="end">{duration(tick, true)}</text>
    </g>)}
    {window.years.map(year => <g key={year} className="grid-line">
      <text x={x(time(`${year}-01-01`))} y={H - 16} textAnchor="middle">{year}</text>
    </g>)}
    {!compact && maximum > reliableRange && <g className="chart-limit">
      <rect x={left} y={top} width={W - right - left} height={Math.max(0, y(reliableRange) - top)}
        fill={`url(#${title}-hatch)`} />
      <line x1={left} x2={W - right} y1={y(reliableRange)} y2={y(reliableRange)} />
      <text x={left + 10} y={y(reliableRange) - 9}>
        Above {duration(reliableRange, true)}: unreliable{mobile ? '' : ' with this task suite'}
      </text>
    </g>}
    <defs><pattern id={`${title}-hatch`} width="9" height="9" patternUnits="userSpaceOnUse">
      <path d="M-2 2 2-2M0 9 9 0M7 11 11 7" stroke="currentColor" strokeWidth="0.5" opacity=".1" />
    </pattern></defs>
    <path d={path} className="frontier-line" />
    {!compact && <g className="interval-line">
      <line x1={x(time(current.release_date))} x2={x(time(current.release_date))}
        y1={y(current[reliability].ci_low)} y2={y(current[reliability].ci_high)} />
      {[current[reliability].ci_low, current[reliability].ci_high].map(v =>
        <line key={v} x1={x(time(current.release_date)) - 5} x2={x(time(current.release_date)) + 5}
          y1={y(v)} y2={y(v)} />)}
    </g>}
    {visible.map(model => {
      const active = model.id === current.id
      return <g key={model.id} className={`chart-point ${active ? 'selected' : ''}`}
        role="button" tabIndex={active ? 0 : -1}
        aria-label={`${model.name}: ${duration(model[reliability].estimate)}`}
        aria-pressed={active} onClick={() => onSelect(model.id)}
        onKeyDown={event => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault(); onSelect(model.id)
          }
        }}>
        <circle className="hit-target" cx={x(time(model.release_date))}
          cy={y(model[reliability].estimate)} r="15" />
        {active && <circle className="selection-ring" cx={x(time(model.release_date))}
          cy={y(model[reliability].estimate)} r="12" />}
        <circle className="point-dot" cx={x(time(model.release_date))}
          cy={y(model[reliability].estimate)} r={active ? 5 : 3.5} />
      </g>
    })}
    {!compact && <text className="axis-caption" x={left} y={H - 1}>
      {mobile ? 'Model release date · log scale' : 'Model release date · human task time, logarithmic scale'}
    </text>}
  </svg>
}
