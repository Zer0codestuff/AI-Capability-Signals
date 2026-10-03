import { useMemo, useState } from 'react'
import type { ExplorerRow } from '../types'
import { Reveal, Segment, MagnifyingGlass } from './ui'
import { compact, day, money, power } from '../lib/format'

type Key = 'e' | 'd' | 'p' | 'params' | 'compute' | 'n'
type Access = 'all' | 'open' | 'closed'

const COLUMNS: { key: Key; label: string; numeric: boolean }[] = [
  { key: 'n', label: 'Model', numeric: false },
  { key: 'd', label: 'Released', numeric: false },
  { key: 'e', label: 'Capability index', numeric: true },
  { key: 'p', label: 'Price per million tokens', numeric: true },
  { key: 'params', label: 'Parameters', numeric: true },
  { key: 'compute', label: 'Training compute', numeric: true },
]

export default function Explorer({ rows }: { rows: ExplorerRow[] }) {
  const [query, setQuery] = useState('')
  const [access, setAccess] = useState<Access>('all')
  const [sort, setSort] = useState<{ key: Key; down: boolean }>({ key: 'e', down: true })

  const shown = useMemo(() => {
    const needle = query.trim().toLowerCase()
    const filtered = rows.filter(row =>
      (access === 'all' || row.a === access)
      && (!needle || row.n.toLowerCase().includes(needle) || row.o.toLowerCase().includes(needle)))
    return filtered.sort((a, b) => {
      const left = a[sort.key]
      const right = b[sort.key]
      if (left === null) return 1
      if (right === null) return -1
      const order = left < right ? -1 : left > right ? 1 : 0
      return sort.down ? -order : order
    })
  }, [rows, query, access, sort])

  return <section className="signal" id="models">
    <Reveal as="header" className="signal-head">
      <p className="eyebrow"><span className="number">11</span>Every model</p>
      <h2>Look up a model</h2>
      <p className="answer">All {rows.length} models in the capability index, with the price, size and training
        compute that could be matched to each one. Empty cells are unknown, not zero.</p>
    </Reveal>
    <Reveal className="card table-card">
      <div className="explorer-controls">
        <label className="search"><MagnifyingGlass size={16} aria-hidden="true" />
          <span className="visually-hidden">Search by model or lab</span>
          <input type="search" value={query} placeholder="Search a model or a lab"
            onChange={event => setQuery(event.target.value)} /></label>
        <Segment label="Access" value={access} onChange={setAccess}
          options={[{ value: 'all', label: 'All' }, { value: 'open', label: 'Open' }, { value: 'closed', label: 'Closed' }]} />
        <span className="count mono" role="status">{shown.length} models</span>
      </div>
      <div className="table-scroll tall"><table>
        <thead><tr>{COLUMNS.map(column => <th key={column.key} className={column.numeric ? 'num' : ''}
          aria-sort={sort.key === column.key ? (sort.down ? 'descending' : 'ascending') : 'none'}>
          <button type="button" onClick={() => setSort(current =>
            ({ key: column.key, down: current.key === column.key ? !current.down : column.key !== 'n' }))}>
            {column.label}<span aria-hidden="true">{sort.key === column.key ? (sort.down ? ' ↓' : ' ↑') : ''}</span>
          </button></th>)}</tr></thead>
        <tbody>{shown.map(row => <tr key={row.n}>
          <th scope="row">{row.n}<small>{row.o} · {row.a === 'other' ? 'access unclear' : row.a}</small></th>
          <td className="mono">{day(row.d, 'month')}</td>
          <td className="num mono">{row.e.toFixed(1)}</td>
          <td className="num mono">{row.p === null ? '' : money(row.p)}</td>
          <td className="num mono">{row.params === null ? '' : compact(row.params)}</td>
          <td className="num mono">{row.compute === null ? '' : power(row.compute)}</td>
        </tr>)}</tbody>
      </table></div>
      {shown.length === 0 && <p className="empty">No model matches that search.</p>}
    </Reveal>
  </section>
}
