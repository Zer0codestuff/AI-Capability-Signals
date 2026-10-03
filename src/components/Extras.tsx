import type { BoardRow, ChartData, Story } from '../types'
import Chart from './Chart'
import { Reveal } from './ui'
import { day, percent, span } from '../lib/format'

type Disclosure = Story['chapters']['size']['bars']['disclosure']

/* Share of notable language models whose size and training compute are public, by year. */
export function DisclosureBars({ rows }: { rows: Disclosure }) {
  return <Reveal className="card bars-card">
    <div className="chart-top">
      <p className="chart-caption">Share of notable language models with a known figure, by release year</p>
      <ul className="legend">
        <li><i className="key-box" style={{ background: 'var(--accent)' }} />Size</li>
        <li><i className="key-box" style={{ background: 'var(--series-2)' }} />Training compute</li>
      </ul>
    </div>
    <div className="bars" role="img" aria-label="Bar chart of disclosure by year. The table below holds the same values.">
      {rows.map(row => <div className="bar-group" key={row.label}>
        <div className="bar-pair">
          <span className="bar" style={{ height: `${row.params * 100}%`, background: 'var(--accent)' }}
            title={`${row.label}: size known for ${percent(row.params)} of ${row.n} models`} />
          <span className="bar" style={{ height: `${row.compute * 100}%`, background: 'var(--series-2)' }}
            title={`${row.label}: compute known for ${percent(row.compute)} of ${row.n} models`} />
        </div>
        <span className="bar-value mono">{percent(row.params)}</span>
        <span className="bar-label mono">{row.label}</span>
      </div>)}
    </div>
    <div className="visually-hidden"><table>
      <caption>Disclosure by year</caption>
      <thead><tr><th>Year</th><th>Models</th><th>Size published</th><th>Compute published</th></tr></thead>
      <tbody>{rows.map(row => <tr key={row.label}><th scope="row">{row.label}</th><td>{row.n}</td>
        <td>{percent(row.params)}</td><td>{percent(row.compute)}</td></tr>)}</tbody>
    </table></div>
  </Reveal>
}

export function LagCard({ chart, today, title }: { chart: ChartData; today: string; title: string }) {
  return <Reveal className="card chart-card slim">
    <div className="chart-top"><p className="chart-caption">{title}</p></div>
    <Chart chart={chart} today={today} projection={false} label={title} slim />
  </Reveal>
}

const COUNTRY: Record<string, string> = { us: 'United States', china: 'China', other: 'Other' }

export function Board({ rows, tracked }: { rows: BoardRow[]; tracked: number }) {
  const best = rows[0].v
  const floor = Math.min(...rows.map(row => row.v)) - 4
  return <Reveal className="card table-card">
    <div className="table-scroll"><table>
      <thead><tr><th>Lab</th><th>Best model</th><th>Capability index</th><th className="num">Time at the top</th></tr></thead>
      <tbody>{rows.map(row => <tr key={row.org}>
        <th scope="row">{row.org}<small>{COUNTRY[row.country]}</small></th>
        <td>{row.n}<small>{day(row.d, 'month')} · {row.access === 'open' ? 'open' : row.access === 'closed' ? 'closed' : 'other'}</small></td>
        <td><span className="meter"><span style={{ width: `${((row.v - floor) / (best - floor)) * 100}%` }} /></span>
          <span className="mono">{row.v.toFixed(1)}</span></td>
        <td className="num mono">{row.days_on_top ? `${span(row.days_on_top / 30.4375)} · ${percent(row.days_on_top / tracked)}` : 'never'}</td>
      </tr>)}</tbody>
    </table></div>
  </Reveal>
}
