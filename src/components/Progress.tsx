import { useState } from 'react'
import type { Reliability, StoryData } from '../types'
import { duration, formatDate } from '../lib/math'
import HorizonChart from './HorizonChart'
import { ArrowUpRight, Evidence, Field, Reveal, SectionHead, Segment, SourceLine } from './ui'

export default function Progress({ data }: { data: StoryData }) {
  const [reliability, setReliability] = useState<Reliability>('p50')
  const [selected, setSelected] = useState(data.trends.p50.anchor_id)
  const model = data.horizons.find(m => m.id === selected)!
  const interval = model[reliability]
  const rangeWarning = interval.estimate > data.benchmark.reliable_range_minutes
  return <section id="progress" className="chapter">
    <SectionHead number="01" label="The progress" title={<>From minutes.<br />To hours.</>}>
      Instead of asking how “smart” AI is, ask how long a task it can handle,
      measured in the time a human expert would need.
    </SectionHead>
    <Reveal className="card chart-card" delay={80}>
      <div className="chart-toolbar">
        <span className="chart-label"><span className="dot" />Measured task horizon</span>
        <Segment value={reliability} onChange={setReliability} label="Required success rate" options={[
          { value: 'p50', label: '50% success' }, { value: 'p80', label: '80% success' },
        ]} />
      </div>
      <div className="measurement-strip" aria-live="polite">
        <div><span className="stat-label">{model.name}</span>
          <strong className="measurement">{duration(interval.estimate)}</strong></div>
        <div className="measurement-context"><span>Human task duration, not AI running time.</span>
          <span>{reliability === 'p50' ? 'Half' : 'About 8 in 10'} of comparable benchmark tasks succeed.</span></div>
        <div className="interval-detail">
          <span>95% interval</span><strong className="mono">{duration(interval.ci_low)} to {duration(interval.ci_high)}</strong>
          <span>{formatDate(model.release_date, true)} release</span>
        </div>
      </div>
      <HorizonChart models={data.horizons} reliability={reliability} selected={selected}
        onSelect={setSelected} reliableRange={data.benchmark.reliable_range_minutes} />
      <div className="chart-bottom">
        <Field label="Explore a model">
          <select value={selected} onChange={event => setSelected(event.target.value)}>
            {data.horizons.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
          </select>
        </Field>
        <p className={`chart-hint ${rangeWarning ? 'is-warning' : ''}`}>{rangeWarning
          ? 'This point exceeds METR’s reliable measurement range. Do not read it as a precise result.'
          : 'Select a dot. Switch to 80% success and watch the task horizon shrink.'}</p>
      </div>
      <SourceLine href="https://metr.org/time-horizons/" label="METR · TH 1.1">
        {data.horizons.length} models · latest model release {formatDate(data.horizons.at(-1)!.release_date, true)}
      </SourceLine>
    </Reveal>
    <Reveal className="evidence-group" delay={120}>
      <Evidence>
        <p>This is an estimated success threshold on METR’s software, machine learning and cybersecurity
          tasks. It does not measure every kind of work, how long an AI runs, or whether you can
          delegate a job without supervision. The line connects successive record high measurements,
          not every release. The vertical bar is the selected model’s published 95% interval.</p>
        <p>Only Time Horizon 1.1 results are included. Three older TH 1.0 records embedded in the source
          are excluded. Agents use different software and tool configurations. METR warns that values
          above 16 hours are unreliable with its current task suite.</p>
        <p><a href="https://metr.org/notes/2026-01-22-time-horizon-limitations/" target="_blank"
          rel="noreferrer">Read METR’s explanation of these limits<ArrowUpRight size={14} aria-hidden="true" /></a></p>
      </Evidence>
      <Evidence title="See every measurement">
        <div className="table-scroll" tabIndex={0} role="region" aria-label="All METR measurements">
          <table><thead><tr><th>Model</th><th>Released</th><th>50% success</th><th>80% success</th></tr></thead>
            <tbody>{data.horizons.map(m => <tr key={m.id}><th scope="row">{m.name}</th>
              <td className="mono">{m.release_date}</td>
              <td className="mono">{duration(m.p50.estimate)}<small>{duration(m.p50.ci_low)} to {duration(m.p50.ci_high)}</small></td>
              <td className="mono">{duration(m.p80.estimate)}<small>{duration(m.p80.ci_low)} to {duration(m.p80.ci_high)}</small></td></tr>)}</tbody>
          </table>
        </div>
      </Evidence>
    </Reveal>
  </section>
}
