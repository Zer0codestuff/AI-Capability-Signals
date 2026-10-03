import type { Story } from '../types'
import { Reveal, ArrowUpRight, DownloadSimple } from './ui'
import { compact, day } from '../lib/format'

const STEPS = [
  { title: 'Pick the frontier', text: 'A trend is fitted on the models that led at the time of their release: record setters, or the ten largest so far. Membership never depends on later releases.' },
  { title: 'Draw one straight line', text: 'For anything that grows by multiplication the line is fitted on a logarithmic scale, where a straight line means steady exponential change.' },
  { title: 'Ask if the pace changed', text: 'The data is split where two lines fit best and the two slopes are compared. The search is repeated on 2,000 resamples, and a change counts only if 95% of them agree.' },
  { title: 'Test it on the past', text: 'The same method is rerun on earlier dates and its one year projection is compared with what was really released. A projection is drawn only if it beat assuming no change.' },
]

export default function Method({ story }: { story: Story }) {
  const { quality } = story
  return <section className="signal" id="method">
    <Reveal as="header" className="signal-head">
      <p className="eyebrow"><span className="number">12</span>Method and sources</p>
      <h2>How the numbers are made</h2>
      <p className="answer">One method for every chart, so they can be compared. No number on this page is typed
        by hand: a script downloads public datasets, computes everything and refuses to publish if a check fails.</p>
    </Reveal>

    <Reveal className="steps">
      {STEPS.map((step, i) => <div className="card step" key={step.title}>
        <span className="number mono">{i + 1}</span><h3>{step.title}</h3><p>{step.text}</p>
      </div>)}
    </Reveal>

    <Reveal className="explain">
      <div><h3>What the data can and cannot say</h3>
        <p>The shaded ranges continue the past. They do not know about new ideas, chip shortages, laws or
          money running out. Where the method failed its own test, no projection is shown at all.</p>
        <p>Model size, compute and cost are inputs. Only the capability index and the task length measure
          what models can do, and both are test results, not a definition of intelligence.</p></div>
      <div><h3>Data notes for this update</h3><ul>
        <li>{quality.database_models.toLocaleString('en-US')} models in the Epoch AI database, {quality.indexed_models} in
          the capability index, {quality.priced_models} matched to a price.</li>
        {quality.corrections.filter(item => item.applied).map(item =>
          <li key={item.model}>Corrected {item.model}: {item.field.toLowerCase()} read {compact(item.from)} in the
            source, used {compact(item.to)}. {item.reason}</li>)}
        <li>Rows dated after the download day are ignored.</li>
      </ul></div>
    </Reveal>

    <Reveal className="card table-card" >
      <div className="table-scroll"><table id="sources">
        <thead><tr><th>Source</th><th>Used for</th><th>License</th><th>Downloaded</th></tr></thead>
        <tbody>{story.sources.map(source => <tr key={source.id}>
          <th scope="row"><a href={source.page} target="_blank" rel="noreferrer">{source.name}
            <ArrowUpRight size={14} aria-hidden="true" /></a><small>{source.publisher}</small></th>
          <td className="wrap">{source.description}</td>
          <td className="wrap">{source.license}</td>
          <td className="mono">{day(source.retrieved_at.slice(0, 10))}
            <small title={source.files.map(file => file.sha256).join('\n')}>sha256 {source.files[0].sha256.slice(0, 10)}</small></td>
        </tr>)}</tbody>
      </table></div>
    </Reveal>

    <Reveal className="closing">
      <a className="button primary" href="./data/signals.json" download><DownloadSimple size={18} aria-hidden="true" />Download the data</a>
      <a className="button ghost" href="https://github.com/Zer0codestuff/AI-Capability-Signals" target="_blank" rel="noreferrer">
        Read the code<ArrowUpRight size={16} aria-hidden="true" /></a>
    </Reveal>
  </section>
}
