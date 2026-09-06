import { useMemo, useState } from 'react'
import type { Score, StoryData } from '../types'
import { byName, byScore, frontier, logScale, linear, money, pricePoints } from '../lib/math'
import { ArrowRight, ArrowUpRight, Evidence, Field, Reveal, SectionHead, Segment, SourceLine } from './ui'
import { useCompactChart } from '../lib/useCompactChart'

const WORKLOADS = [
  { name: 'Short exchange', input: 1000, output: 500, detail: '1,000 input + 500 output tokens' },
  { name: 'Document analysis', input: 16000, output: 2000, detail: '16,000 input + 2,000 output tokens' },
  { name: 'Long context', input: 128000, output: 4000, detail: '128,000 input + 4,000 output tokens' },
] as const

export default function Price({ data }: { data: StoryData }) {
  const mobile = useCompactChart()
  const [metric, setMetric] = useState<Score>('intelligence')
  const [workload, setWorkload] = useState(0)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [budget, setBudget] = useState(10)
  const work = WORKLOADS[workload]
  const points = useMemo(() => pricePoints(data.prices, metric, work.input, work.output),
    [data.prices, metric, work])
  const efficient = useMemo(() => frontier(points), [points])
  const best = [...points].sort(byScore)[0]
  const selected = points.find(p => p.id === selectedId) ?? best
  const toolbar = <div className="chart-toolbar">
    <Field label="Workload">
      <select value={workload} onChange={e => setWorkload(Number(e.target.value))}>
        {WORKLOADS.map((w, i) => <option key={w.name} value={i}>{w.name}</option>)}
      </select>
    </Field>
    <Segment value={metric} onChange={setMetric} label="Benchmark category" options={[
      { value: 'intelligence', label: 'General' }, { value: 'coding', label: 'Coding' },
      { value: 'agentic', label: 'Agents' },
    ]} />
  </div>
  if (!best) return <section id="cost" className="chapter">
    <SectionHead number="02" label="The cost" title="No models fit this selection.">
      No model in this snapshot has both a usable score and enough context for the selected workload.
      Try a shorter workload or another benchmark.
    </SectionHead>
    <div className="card chart-card">{toolbar}
      <p className="empty-note">Change the workload or the benchmark category above to bring models back.</p></div>
  </section>
  const W = mobile ? 600 : 1040, H = 430, left = 56, right = 26, top = 26, bottom = H - 62
  const maxPrice = Math.max(...points.map(p => p.cost), 1)
  const minPrice = Math.min(...points.map(p => p.cost), 0.1)
  const lowPower = Math.floor(Math.log10(minPrice))
  const highPower = Math.ceil(Math.log10(maxPrice))
  const x = logScale([10 ** lowPower, 10 ** highPower], [left, W - right])
  const maxScore = Math.ceil(Math.max(...points.map(p => p.score)) / 20) * 20
  const y = linear([0, maxScore], [bottom, top])
  const budgetPick = [...points].filter(p => p.cost <= budget)
    .sort(byScore)[0]
  const visibleList = [...points].filter(p => `${p.name} ${p.provider}`.toLowerCase().includes(query.toLowerCase()))
    .sort(byScore)
  const efficientIds = new Set(efficient.map(p => p.id))
  const path = efficient.map((p, i) => `${i ? 'L' : 'M'}${x(p.cost)},${y(p.score)}`).join(' ')

  return <section id="cost" className="chapter">
    <SectionHead number="02" label="The cost" title={<>More capable.<br />Not always more expensive.</>}>
      Each dot is a model. Higher means a better benchmark score. Further left means a lower
      list price for the same number of tokens.
    </SectionHead>
    <Reveal className="card chart-card" delay={80}>
      {toolbar}
      <div className="price-selection" aria-live="polite">
        <div><span className="stat-label">Selected model</span><h3>{selected.name}</h3>
          <span className="muted">{selected.provider}</span></div>
        <div><span className="stat-label">Per 1,000 exchanges</span><strong>{money(selected.cost)}</strong></div>
        <div><span className="stat-label">Benchmark index</span><strong>{selected.score.toFixed(1)}<small> / 100</small></strong></div>
        <a className="icon-link" href={selected.url} target="_blank" rel="noreferrer"
          aria-label={`Open ${selected.name} source`}><ArrowUpRight size={18} aria-hidden="true" /></a>
      </div>
      <svg className="data-chart price-chart" viewBox={`0 0 ${W} ${H}`} role="group"
        aria-label="Benchmark score versus cost per 1,000 exchanges. The table below lists all models."
        onKeyDown={event => {
          if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
            event.preventDefault()
            const index = points.findIndex(p => p.id === selected.id)
            setSelectedId(points[(index + (event.key === 'ArrowRight' ? 1 : -1) + points.length) % points.length].id)
          }
        }}>
        {Array.from({ length: maxScore / 20 + 1 }, (_, i) => i * 20).map(tick =>
          <g className="grid-line" key={tick}>
            <line x1={left} x2={W - right} y1={y(tick)} y2={y(tick)} />
            <text x={left - 16} y={y(tick) + 4} textAnchor="end">{tick}</text>
          </g>)}
        {Array.from({ length: highPower - lowPower + 1 }, (_, i) => 10 ** (i + lowPower)).map(tick =>
          <g className="grid-line" key={tick}>
            <text x={x(tick)} y={mobile ? H - 38 : H - 32} textAnchor="middle">${tick < 1 ? tick.toFixed(2) : tick.toLocaleString('en')}</text>
          </g>)}
        <text className="axis-caption" x={left + 12} y={top + 8}>BETTER SCORE ↑</text>
        <text className="axis-caption" x={W - right} y={H - 8} textAnchor="end">
          ← LOWER COST · USD, logarithmic scale
        </text>
        <path d={path} className="frontier-line subdued" />
        {points.map(point => <g key={point.id}
          className={`chart-point ${selected.id === point.id ? 'selected' : ''} ${efficientIds.has(point.id) ? 'efficient' : ''}`}
          role="button" tabIndex={selected.id === point.id ? 0 : -1}
          aria-pressed={selected.id === point.id}
          aria-label={`${point.name}, index ${point.score}, ${money(point.cost)}`}
          onClick={() => setSelectedId(point.id)}
          onKeyDown={e => {
            if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setSelectedId(point.id) }
          }}>
          <circle className="hit-target" cx={x(point.cost)} cy={y(point.score)} r="12" />
          {selected.id === point.id && <circle className="selection-ring" cx={x(point.cost)} cy={y(point.score)} r="12" />}
          <circle className="point-dot" cx={x(point.cost)} cy={y(point.score)}
            r={selected.id === point.id ? 5.5 : efficientIds.has(point.id) ? 4 : 3} />
        </g>)}
      </svg>
      <div className="chart-bottom"><span className="chart-label"><span className="dot" />Best score for the money</span>
        <Field label="Select a model">
          <select value={selected.id} onChange={e => setSelectedId(e.target.value)}>
            {[...points].sort(byName).map(p =>
              <option key={p.id} value={p.id}>{p.name}</option>)}
          </select>
        </Field>
      </div>
      <SourceLine href="https://openrouter.ai/models" label="OpenRouter + Artificial Analysis">
        {points.length} models · {work.detail} per exchange · current prices, not price history
      </SourceLine>
    </Reveal>
    <Reveal className="card budget-card" delay={120}>
      <label htmlFor="budget">What could <strong className="accent">${budget}</strong> buy?
        <small>Budget for 1,000 {work.name.toLowerCase()}s</small></label>
      <input id="budget" type="range" min="1" max="100" value={budget}
        onChange={e => setBudget(Number(e.target.value))} aria-valuetext={`${budget} US dollars`} />
      <div className="budget-result" aria-live="polite">
        {budgetPick ? <><button type="button" onClick={() => setSelectedId(budgetPick.id)}>{budgetPick.name}<ArrowRight size={16} aria-hidden="true" /></button>
          <small>Highest listed index within budget · {money(budgetPick.cost)}</small></>
          : <><strong>No matching model</strong><small>Try a larger budget or a shorter workload.</small></>}
      </div>
    </Reveal>
    <Reveal className="evidence-group" delay={160}>
      <Evidence title="Prices are an estimate of a bill, not a promise">
        <p>Cost = 1,000 × [input tokens × input rate + output tokens × output rate]. The workload is
          a declared token budget, not a measured task. It excludes caching, tax, tool fees, retries,
          and additional reasoning tokens. Different models can use different token counts for the
          same task. OpenRouter’s listed route may change.</p>
        <p>Context length tiers apply where published. Models that cannot fit the chosen workload are
          omitted. Free quotas, special routing variants, non text outputs, and models without usable
          prices or benchmarks are excluded. {data.price_coverage.included} of {data.price_coverage.catalogue_rows}
          {' '}catalogue rows are included before workload filtering.</p>
        <p>{data.price_coverage.benchmark_note} “General” is the Artificial Analysis Intelligence Index;
          coding and agent scores are separate indices. Their scales are not interchangeable and a
          score of 60 does not mean 60% of arbitrary tasks succeed.</p>
        <p><a href="https://artificialanalysis.ai/methodology/intelligence-benchmarking" target="_blank"
          rel="noreferrer">Read the benchmark methodology<ArrowUpRight size={14} aria-hidden="true" /></a></p>
      </Evidence>
      <Evidence title="Search the model catalogue">
        <label className="search-label" htmlFor="model-search">Model or provider</label>
        <input id="model-search" className="search-input" type="search" placeholder="GPT, Claude, Qwen"
          value={query} onChange={e => setQuery(e.target.value)} />
        <p className="muted" aria-live="polite">{visibleList.length} matching models</p>
        {visibleList.length ? <div className="table-scroll catalogue-table" role="region" aria-label="Model catalogue" tabIndex={0}>
          <table><thead><tr><th>Model</th><th>Index</th><th>Cost / 1,000</th><th>Source</th></tr></thead>
            <tbody>{visibleList.map(p => <tr key={p.id}><th scope="row">
              <button type="button" onClick={() => setSelectedId(p.id)}>{p.name}</button></th>
              <td className="mono">{p.score.toFixed(1)}</td><td className="mono">{money(p.cost)}</td>
              <td><a href={p.url} target="_blank" rel="noreferrer" aria-label={`Source for ${p.name}`}><ArrowUpRight size={16} aria-hidden="true" /></a></td>
            </tr>)}</tbody></table>
        </div> : <p className="empty-note">No model or provider matches “{query}”. Try a shorter name.</p>}
      </Evidence>
    </Reveal>
  </section>
}
