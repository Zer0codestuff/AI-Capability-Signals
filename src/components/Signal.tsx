import { useState, type ReactNode } from 'react'
import type { ChartData, Trend } from '../types'
import Chart, { seriesColor } from './Chart'
import { Fold, Reveal, Segment, ArrowUpRight } from './ui'
import { amount, day, doubling, miss, pace, paceRange, percent, UNIT_LABEL } from '../lib/format'

export interface SignalCopy {
  id: string
  eyebrow: string
  question: string
  answer: ReactNode
  means: ReactNode[]
  caveats: ReactNode[]
  method: ReactNode
  source: { label: string; href: string; note: string }
  caption?: string
  pick?: boolean
  /** Replaces the trend readouts and hides the trend line, for charts that compare two groups. */
  tiles?: Readout[]
}

export interface Readout { label: string; value: string; note: ReactNode }

const VERDICT = {
  speeding_up: 'Speeding up',
  slowing_down: 'Slowing down',
  steady: 'No clear change',
  unknown: 'Too early to tell',
} as const

function readouts(trend: Trend): Readout[] {
  const list: Readout[] = []
  const double = doubling(trend)
  const basis = trend.basis === 'recent' ? `since ${day(trend.line[0].d, 'month')}` : `since ${day(trend.window.from, 'year')}`
  list.push({
    label: 'Pace',
    value: pace(trend, trend.log),
    note: <>{double ? `The value ${double}. ` : ''}Fitted on {trend.n} points {basis}. 90% range: {paceRange(trend)}.</>,
  })
  const { shape } = trend
  let note: ReactNode
  if (shape.verdict === 'unknown') note = 'There are not enough points over enough years to compare two periods.'
  else if (shape.early && shape.late && shape.split) {
    const before = pace(shape.early, trend.log)
    const after = pace(shape.late, trend.log)
    note = shape.verdict === 'steady'
      ? <>Splitting the data at {day(shape.split, 'month')} gives {before} before and {after} after. With this few points the difference could be chance.</>
      : <>{before} before {day(shape.split, 'month')}, {after} since. The projection uses the recent pace.</>
  }
  list.push({ label: 'Is the pace changing?', value: VERDICT[shape.verdict], note })
  const check = trend.backtest
  if (check) {
    const better = check.typical_error < check.naive_error
    list.push({
      label: 'Tested on the past',
      value: better ? `Off by ${miss(check, check.typical_error)}` : 'Failed the test',
      note: <>Projecting a year ahead from {check.cutoffs} earlier dates, the typical miss was {miss(check, check.typical_error)}.
        Assuming no change at all missed by {miss(check, check.naive_error)}.
        {better ? <> The real value fell inside the shaded range {percent(check.coverage)} of the time.</> : <> So no projection is drawn.</>}</>,
    })
  } else {
    list.push({ label: 'Tested on the past', value: 'Not possible yet', note: 'Too few points to rerun the method on earlier dates, so no projection is drawn.' })
  }
  return list
}

export function Legend({ chart, projection, trend }: { chart: ChartData; projection: boolean; trend: boolean }) {
  const groups = chart.groups.filter(group => chart.points.some(point => point.g === group.id)
    && !chart.series.some(series => series.id === group.id))
  return <ul className="legend">
    {chart.series.map(series => <li key={series.id}>
      <i className={series.kind === 'points' ? 'key-dot' : 'key-line'} style={{ background: seriesColor(chart, series.id) }} />{series.label}</li>)}
    {groups.map(group => <li key={group.id}>
      <i className="key-dot" style={{ background: group.id === 'frontier' ? 'var(--accent)' : 'var(--dot)' }} />{group.label}</li>)}
    {trend && chart.series.some(series => series.trend) && <li><i className="key-line fit-key" />Trend</li>}
    {projection && <li><i className="key-band" />If the trend continues, 80% range</li>}
  </ul>
}

export default function Signal({ copy, number, chart, today, children }: {
  copy: SignalCopy; number: string; chart: ChartData; today: string; children?: ReactNode
}) {
  const fitted = chart.series.filter(series => series.trend)
  const [active, setActive] = useState(fitted[0]?.id ?? chart.series[0]?.id)
  const [wanted, setWanted] = useState(true)
  const trend = copy.tiles ? null : chart.series.find(series => series.id === active)?.trend ?? null
  const projection = wanted && !!trend?.projectable
  const tiles = copy.tiles ?? (trend ? readouts(trend) : [])
  const records = chart.series.find(series => series.id === active) ?? chart.series[0]

  return <section className="signal" id={copy.id}>
    <Reveal as="header" className="signal-head">
      <p className="eyebrow"><span className="number">{number}</span>{copy.eyebrow}</p>
      <h2>{copy.question}</h2>
      <p className="answer">{copy.answer}</p>
    </Reveal>

    <Reveal className="card chart-card">
      <div className="chart-top">
        <p className="chart-caption">{copy.caption ?? UNIT_LABEL[chart.unit]}{chart.scale === 'log' ? ' · each gridline is a fixed multiple' : ''}</p>
        <div className="chart-controls">
          {copy.pick && fitted.length > 1 && <Segment label="Level shown as a trend" value={active}
            options={fitted.map(series => ({ value: series.id, label: series.label }))} onChange={setActive} />}
          {trend?.projectable && <button type="button" className="toggle" aria-pressed={projection}
            onClick={() => setWanted(value => !value)}><span className="switch" aria-hidden="true" />If the trend continues</button>}
        </div>
      </div>
      <Chart chart={chart} today={today} projection={projection} active={active} trend={!copy.tiles}
        label={`${copy.question} Chart of ${UNIT_LABEL[chart.unit].toLowerCase()} over time.`} />
      <Legend chart={chart} projection={projection} trend={!copy.tiles} />
      <p className="source-line"><span>{copy.source.note}</span>
        <a href={copy.source.href} target="_blank" rel="noreferrer">{copy.source.label}<ArrowUpRight size={14} aria-hidden="true" /></a></p>
    </Reveal>

    {tiles.length > 0 && <Reveal className="readouts">
      {tiles.map(tile => <div className="card readout" key={tile.label}>
        <span className="stat-label">{tile.label}</span>
        <strong>{tile.value}</strong>
        <p>{tile.note}</p>
      </div>)}
    </Reveal>}

    {projection && trend && trend.milestones.length > 0 && <Reveal className="card milestones">
      <span className="stat-label">If the trend continues</span>
      <ul>{trend.milestones.map(milestone => <li key={milestone.id}>
        <strong>{day(milestone.d, 'month')}</strong>
        <span>{milestone.label}</span>
        <small>80% range {day(milestone.lo, 'month')} to {day(milestone.hi, 'month')}</small>
      </li>)}</ul>
    </Reveal>}

    {children}

    <Reveal className="explain">
      <div><h3>What this means</h3>{copy.means.map((text, i) => <p key={i}>{text}</p>)}</div>
      <div><h3>Keep in mind</h3><ul>{copy.caveats.map((text, i) => <li key={i}>{text}</li>)}</ul></div>
    </Reveal>

    <div className="folds">
      <Fold title="Method">{<div className="prose">{copy.method}</div>}</Fold>
      {records && <Fold title={`Data table: ${records.label}`}>
        <div className="table-scroll short"><table>
          <thead><tr><th>Date</th><th>Name</th><th className="num">{UNIT_LABEL[chart.unit]}</th></tr></thead>
          <tbody>{[...records.points].reverse().map((point, i) => <tr key={i}>
            <td className="mono">{day(point.d)}</td><th scope="row">{point.n}</th>
            <td className="num mono">{amount(chart.unit, point.v)}</td></tr>)}</tbody>
        </table></div>
      </Fold>}
    </div>
  </section>
}
