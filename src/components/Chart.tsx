import { useEffect, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react'
import type { ChartData, Point, Series, Unit } from '../types'
import { amount, day, time, TIME_TICKS } from '../lib/format'
import { linearScale, logScale, tickedLogScale, yearTicks } from '../lib/scale'

/* Fixed order, validated together on the card surface: lime, blue, magenta, yellow.
   A series keeps its colour wherever it appears. */
export const SERIES_COLORS = ['var(--accent)', 'var(--series-2)', 'var(--series-3)', 'var(--series-4)']

const MARGIN = { top: 20, right: 20, bottom: 30, left: 60 }
const HOVER_RADIUS = 36

interface Mark { point: Point; x: number; y: number; color: string; series?: string; strong: boolean }

function useWidth() {
  const ref = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(0)
  useEffect(() => {
    const node = ref.current
    if (!node) return
    setWidth(node.clientWidth)
    if (typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver(entries => setWidth(Math.round(entries[0].contentRect.width)))
    observer.observe(node)
    return () => observer.disconnect()
  }, [])
  return [ref, width] as const
}

export function seriesColor(chart: ChartData, id: string) {
  const index = chart.series.findIndex(series => series.id === id)
  return SERIES_COLORS[Math.max(0, index) % SERIES_COLORS.length]
}

function groupColor(chart: ChartData, group: string | undefined) {
  if (group === 'frontier') return SERIES_COLORS[0]
  if (group && chart.series.some(series => series.id === group)) return seriesColor(chart, group)
  return 'var(--dot)'
}

function detail(unit: Unit, point: Point) {
  const lines: string[] = []
  if (point.lo !== undefined && point.hi !== undefined) {
    lines.push(`Published range ${amount(unit, point.lo)} to ${amount(unit, point.hi)}`)
  }
  if (point.p80 !== undefined) lines.push(`${amount(unit, point.p80)} at 80% success`)
  if (point.e !== undefined) lines.push(`Capability index ${point.e.toFixed(1)}`)
  if (point.k) lines.push(point.k === 'observed' ? 'Price recorded at the time' : 'Vendor list price')
  if (point.m) lines.push(`Level first reached by ${point.m}`)
  return lines
}

export default function Chart({ chart, today, projection, active, label, trend: showTrend = true, slim = false }: {
  chart: ChartData
  today: string
  projection: boolean
  active?: string
  label: string
  trend?: boolean
  slim?: boolean
}) {
  const [ref, width] = useWidth()
  const [hover, setHover] = useState<Mark | null>(null)
  const height = Math.round(Math.min(slim ? 300 : 520, Math.max(slim ? 220 : 300, width * (slim ? 0.3 : 0.52))))
  const compact = width < 560

  const model = useMemo(() => {
    if (!width) return null
    const drawn: Series | undefined = chart.series.find(s => s.id === active) ?? chart.series.find(s => s.trend)
    const trend = showTrend ? drawn?.trend ?? null : null
    const band = projection && trend?.projectable ? trend.band : []
    const everything = [...chart.points, ...chart.series.flatMap(s => s.points)]
    const times = everything.map(p => time(p.d))
    const minTime = Math.min(...times)
    const maxTime = band.length ? time(band[band.length - 1].d) : time(today)
    const left = compact ? 44 : MARGIN.left
    const plot = { left, right: width - MARGIN.right, top: MARGIN.top, bottom: height - MARGIN.bottom }
    const padTime = (maxTime - minTime) * 0.02
    const x = (t: number) =>
      plot.left + ((t - minTime + padTime) / (maxTime - minTime + padTime * 2)) * (plot.right - plot.left)

    const values = everything.map(p => p.v)
    for (const p of chart.points) if (p.g === 'frontier' && p.lo && p.hi) values.push(p.lo, p.hi)
    if (trend) values.push(...trend.line.map(p => p.v))
    for (const step of band) values.push(step.lo, step.hi)
    const low = Math.min(...values)
    const high = Math.max(...values)
    const y = chart.unit === 'minutes'
      ? tickedLogScale(low, high, plot.bottom, plot.top, TIME_TICKS)
      : chart.scale === 'log'
      ? logScale(low, high, plot.bottom, plot.top, compact ? 5 : 7)
      : linearScale(low - (high - low) * 0.04, high + (high - low) * 0.06, plot.bottom, plot.top, compact ? 4 : 6)

    const strongGroups = new Set(['frontier', ...chart.series.map(s => s.id)])
    const marks: Mark[] = chart.points.map(point => ({
      point, x: x(time(point.d)), y: y(point.v), color: groupColor(chart, point.g), strong: strongGroups.has(point.g ?? ''),
    }))
    // A series point usually is one of the dots: show the dot's fuller details on hover.
    const full = new Map(chart.points.map(point => [`${point.n}|${point.d}`, point]))
    const seriesMarks: Mark[] = chart.series.flatMap(series => series.points.map(point => ({
      point: { ...full.get(`${point.n}|${point.d}`), ...point },
      x: x(time(point.d)), y: y(point.v), color: seriesColor(chart, series.id), series: series.id, strong: true,
    })))

    const paths = chart.series.map(series => {
      const points = series.points.map(p => [x(time(p.d)), y(p.v)] as const)
      let d = ''
      points.forEach(([px, py], i) => {
        if (series.kind === 'points') return
        d += i === 0 ? `M${px},${py}` : series.kind === 'step' ? `H${px}V${py}` : `L${px},${py}`
      })
      if (series.kind === 'step' && points.length) d += `H${x(time(today))}`
      return { id: series.id, d, kind: series.kind, color: seriesColor(chart, series.id), last: series.points[series.points.length - 1] }
    })

    const fit = trend
      ? `M${x(time(trend.line[0].d))},${y(trend.line[0].v)}L${x(time(trend.line[1].d))},${y(trend.line[1].v)}`
      : ''
    const forecast = band.length
      ? {
          mid: band.map((s, i) => `${i ? 'L' : 'M'}${x(time(s.d))},${y(s.v)}`).join(''),
          area: band.map((s, i) => `${i ? 'L' : 'M'}${x(time(s.d))},${y(s.hi)}`).join('')
            + [...band].reverse().map(s => `L${x(time(s.d))},${y(s.lo)}`).join('') + 'Z',
          end: band[band.length - 1],
          color: drawn ? seriesColor(chart, drawn.id) : SERIES_COLORS[0],
        }
      : null

    return {
      plot, x, y, marks, seriesMarks, paths, fit, forecast, drawn,
      years: yearTicks(minTime, maxTime, plot.right - plot.left),
      refs: chart.refs.filter(r => r.v >= y.domain[0] && r.v <= y.domain[1]),
      todayX: x(time(today)),
    }
  }, [chart, width, height, today, projection, active, compact, showTrend])

  function nearest(event: PointerEvent<SVGSVGElement>) {
    if (!model) return
    const box = event.currentTarget.getBoundingClientRect()
    const px = event.clientX - box.left
    const py = event.clientY - box.top
    let best: Mark | null = null
    let bestDistance = HOVER_RADIUS ** 2
    // Highlighted marks win ties so a record is never hidden behind a grey dot.
    for (const mark of [...model.seriesMarks, ...model.marks]) {
      const distance = (mark.x - px) ** 2 + (mark.y - py) ** 2
      if (distance < bestDistance - (mark.strong ? 0 : 60)) { best = mark; bestDistance = distance }
    }
    setHover(best)
  }

  function step(event: KeyboardEvent<HTMLDivElement>) {
    if (!model) return
    if (event.key === 'Escape') { setHover(null); return }
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return
    event.preventDefault()
    const line = model.seriesMarks.filter(mark => mark.series === (model.drawn?.id ?? chart.series[0]?.id))
    if (!line.length) return
    const current = hover ? line.findIndex(mark => mark.point === hover.point) : -1
    const next = current < 0
      ? (event.key === 'ArrowLeft' ? line.length - 1 : 0)
      : Math.min(line.length - 1, Math.max(0, current + (event.key === 'ArrowRight' ? 1 : -1)))
    setHover(line[next])
  }

  return <div className={`chart ${slim ? 'slim' : ''}`} ref={ref} tabIndex={0} role="group" onKeyDown={step} onBlur={() => setHover(null)}
    aria-label={`${label} Use the left and right arrow keys to step through the highlighted points.`}>
    {model && <svg width={width} height={height} role="img" aria-label={label}
      onPointerMove={nearest} onPointerLeave={() => setHover(null)}>
      <g className="grid">
        {model.y.ticks.map(tick => <g key={tick}>
          <line x1={model.plot.left} x2={model.plot.right} y1={model.y(tick)} y2={model.y(tick)} />
          <text x={model.plot.left - 10} y={model.y(tick)} dy="0.32em" textAnchor="end">{amount(chart.unit, tick, true)}</text>
        </g>)}
        {model.years.map(tick => <text key={tick.at} x={model.x(tick.at)} y={height - 8} textAnchor="middle">{tick.label}</text>)}
      </g>

      {model.refs.map(ref_ => <g key={ref_.label} className="ref">
        <line x1={model.plot.left} x2={model.plot.right} y1={model.y(ref_.v)} y2={model.y(ref_.v)} />
        <text x={model.plot.left + 6} y={model.y(ref_.v) - 6}>{ref_.label}</text>
      </g>)}

      {model.forecast && <g className="forecast">
        <line className="today" x1={model.todayX} x2={model.todayX} y1={model.plot.top} y2={model.plot.bottom} />
        <text className="today-label" x={model.todayX - 6} y={model.plot.bottom - 8} textAnchor="end">Today</text>
        <path d={model.forecast.area} fill={model.forecast.color} opacity={0.12} />
        <path className="forecast-mid" d={model.forecast.mid} />
        <text className="end-label" textAnchor="end" x={model.plot.right - 4}
          y={Math.max(model.plot.top + 12, model.y(model.forecast.end.v) - 10)}>
          {amount(chart.unit, model.forecast.end.v)} by {day(model.forecast.end.d, 'month')}
        </text>
      </g>}

      <g>
        {model.marks.filter(mark => !mark.strong).map((mark, i) =>
          <circle key={i} className="dot" cx={mark.x} cy={mark.y} r={compact ? 2 : 2.5} />)}
      </g>
      <g>
        {model.marks.filter(mark => mark.strong && mark.point.lo && mark.point.hi && chart.series.length === 1).map((mark, i) =>
          <line key={i} className="whisker" x1={mark.x} x2={mark.x}
            y1={model.y(mark.point.lo!)} y2={model.y(mark.point.hi!)} stroke={mark.color} />)}
      </g>
      {model.paths.map(path => <path key={path.id} className="series-line" d={path.d} stroke={path.color} />)}
      {model.fit && <path className="fit" d={model.fit} />}
      <g>
        {model.marks.filter(mark => mark.strong).map((mark, i) =>
          <circle key={i} className="dot strong" cx={mark.x} cy={mark.y} r={compact ? 3 : 3.5} style={{ fill: mark.color }} />)}
        {model.seriesMarks.map((mark, i) =>
          <circle key={`s${i}`} className="dot strong" cx={mark.x} cy={mark.y} r={compact ? 3 : 4} style={{ fill: mark.color }} />)}
      </g>

      {!model.forecast && model.paths.filter(path => path.kind !== 'points').map(path => {
        const x = model.x(time(path.last.d))
        const y = model.y(path.last.v)
        const below = chart.series.length > 1 && path.id === chart.series[chart.series.length - 1].id
        return <text key={path.id} className="end-label" textAnchor="end" x={Math.min(x + 4, model.plot.right)}
          y={below ? y + 20 : Math.max(model.plot.top + 2, y - 12)}>{path.last.n === 'Today' ? '' : path.last.n}</text>
      })}

      {hover && <circle className="hover-ring" cx={hover.x} cy={hover.y} r={8} stroke={hover.strong ? hover.color : 'var(--ink)'} />}
    </svg>}

    {hover && model && <div className={`tooltip ${hover.x > width * 0.6 ? 'left' : ''} ${hover.y < 110 ? 'below' : ''}`}
      style={{ left: hover.x, top: hover.y }} role="status">
      <strong>{amount(chart.unit, hover.point.v)}</strong>
      <span className="tooltip-name"><i style={{ background: hover.color }} />{hover.point.n}</span>
      <span>{[hover.point.o, day(hover.point.d)].filter(Boolean).join(' · ')}</span>
      {detail(chart.unit, hover.point).map(line => <span key={line}>{line}</span>)}
    </div>}
  </div>
}
