import { useMemo, useState } from 'react'
import type { StoryData } from '../types'
import { byName, byScore, formatDate, money, pricePoints, projectCost } from '../lib/math'
import { Evidence } from './ui'

export default function CostFuture({ data }: { data: StoryData }) {
  const points = useMemo(() => pricePoints(data.prices, 'intelligence', 16000, 2000)
    .sort(byScore), [data.prices])
  const [modelId, setModelId] = useState(points[0]?.id ?? '')
  const [decline, setDecline] = useState(25)
  const [years, setYears] = useState(1)
  const model = points.find(p => p.id === modelId) ?? points[0]
  if (!model) return <div className="cost-future" role="status"><h3>No priced model fits this workload.</h3>
    <p>The price scenario needs a benchmarked model supporting 16,000 input and 2,000 output tokens.
      The rest of the story remains available.</p></div>
  const estimate = projectCost(model.cost, decline / 100, years)

  return <div className="cost-future">
    <div className="eyebrow"><span className="scenario-symbol">◇</span>A second “what if”</div>
    <div className="cost-future-head"><h3>What if those tokens<br />got cheaper?</h3>
      <p>There is no price-history forecast here. Choose an annual price change and see what
        it would do to the same token budget.</p></div>
    <div className="cost-comparison" aria-live="polite">
      <div><span className="stat-label">Today’s list-price estimate</span>
        <strong>{money(model.cost)}</strong><small>{formatDate(data.as_of, true)}</small></div>
      <span className="cost-arrow" aria-hidden="true">→</span>
      <div><span className="stat-label">If prices fall {decline}% a year</span>
        <strong className="accent">{money(estimate)}</strong><small>After {years} {years === 1 ? 'year' : 'years'}</small></div>
    </div>
    <div className="cost-future-controls">
      <label className="select-label">Reference model<select value={modelId} onChange={e => setModelId(e.target.value)}>
        {[...points].sort(byName).map(p =>
          <option key={p.id} value={p.id}>{p.name}</option>)}
      </select></label>
      <div className="horizon-slider"><label htmlFor="price-drop">Assumed annual price fall <strong>{decline}%</strong></label>
        <input id="price-drop" type="range" min="0" max="75" step="5" value={decline}
          onChange={e => setDecline(Number(e.target.value))} aria-valuetext={`${decline} percent per year`} /></div>
      <label className="select-label">Time<select value={years} onChange={e => setYears(Number(e.target.value))}>
        <option value="1">1 year</option><option value="2">2 years</option><option value="3">3 years</option>
      </select></label>
    </div>
    <p className="cost-future-note">1,000 exchanges · 16,000 input + 2,000 output tokens each.
      Same token volume, not guaranteed equal capability. This is not a price quote for a future GPT.</p>
    <Evidence title="Price scenario assumptions">
      <p>Scenario cost = today’s workload cost × (1 − annual price fall)^years.
        The default 25% is an illustrative assumption, not a rate estimated from historical data.
        Input and output rates fall together; token volume stays fixed. Real spending may instead
        rise if models use more reasoning tokens or you make more requests. Prices can also increase,
        models can be withdrawn, and available capability can change.</p>
      <p>The baseline uses the same tier-aware arithmetic as the cost chart. Caching, tools,
        retries, tax, and additional reasoning tokens are excluded. The reference model is an
        example, not a recommendation.</p>
    </Evidence>
  </div>
}
