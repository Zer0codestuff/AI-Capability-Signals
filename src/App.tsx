import { useEffect, useState } from 'react'
import type { StoryData } from './types'
import { duration, formatDate } from './lib/math'
import { Arrow, Evidence, Mark } from './components/ui'
import HorizonChart from './components/HorizonChart'
import Progress from './components/Progress'
import Price from './components/Price'
import Scale from './components/Scale'
import Future from './components/Future'

function useActiveChapter() {
  const [active, setActive] = useState('progress')
  useEffect(() => {
    const observer = new IntersectionObserver(entries => {
      const visible = entries.filter(entry => entry.isIntersecting)
      if (visible.length) setActive(visible[0].target.id)
    }, { rootMargin: '-15% 0px -65% 0px' })
    document.querySelectorAll('section.chapter').forEach(section => observer.observe(section))
    return () => observer.disconnect()
  }, [])
  return active
}

function Story({ data }: { data: StoryData }) {
  const active = useActiveChapter()
  const [teaserModel, setTeaserModel] = useState(data.trends.p50.anchor_id)
  const first = data.horizons.find(m => m.id === 'gpt_4') ?? data.horizons[0]
  const anchor = data.horizons.find(m => m.id === data.trends.p50.anchor_id)!
  const multiplier = Math.round(anchor.p50.estimate / first.p50.estimate)
  const teaser = data.horizons.find(m => m.id === teaserModel)!
  useEffect(() => {
    const id = window.location.hash.slice(1)
    if (!['top', 'progress', 'cost', 'scale', 'future', 'sources'].includes(id)) return
    const frame = requestAnimationFrame(() =>
      document.getElementById(id)?.scrollIntoView({ behavior: 'instant' }))
    return () => cancelAnimationFrame(frame)
  }, [])
  return <>
    <a className="skip-link" href="#main">Skip to the story</a>
    <header className="site-header">
      <a className="brand" href="#top" aria-label="AI Capability Signals, back to top"><Mark />
        <span>AI CAPABILITY<span className="brand-last"> SIGNALS</span></span></a>
      <nav aria-label="Story chapters">
        {(['progress', 'cost', 'scale', 'future'] as const).map((id, i) =>
          <a key={id} href={`#${id}`} aria-current={active === id ? 'location' : undefined}>
            <span className="nav-number">0{i + 1}</span>{id === 'cost' ? 'Cost' : id[0].toUpperCase() + id.slice(1)}
          </a>)}
      </nav>
      <a className="header-source" href="#sources">Sources <Arrow /></a>
    </header>
    <main id="main">
      <section className="hero" id="top">
        <div className="hero-overline"><span className="eyebrow"><span className="dot" />An independent data story</span>
          <span className="update-date">Data retrieved {formatDate(data.as_of, true)}</span></div>
        <div className="hero-content">
          <div className="hero-copy"><h1>AI is moving fast.<br /><span>How far can it go?</span></h1>
            <p>Real progress. Real prices. Possible futures.<br />
              A closer look at the numbers behind the next generation of AI.</p>
            <a className="primary-link" href="#progress">Follow the evidence <Arrow down /></a>
          </div>
          <div className="hero-signal">
            <span className="stat-label">One measured change</span>
            <strong>{multiplier}<span>×</span></strong>
            <span className="hero-signal-label">longer task horizon</span>
            <p>{duration(first.p50.estimate)} → {duration(anchor.p50.estimate)}<br />
              {first.release_date.slice(0, 4)} to {anchor.release_date.slice(0, 4)} · 50% success</p>
          </div>
        </div>
        <div className="hero-chart">
          <div className="hero-chart-label"><span>HUMAN TASK TIME</span>
            <span aria-live="polite">{teaser.name} · {duration(teaser.p50.estimate)} · METR TH 1.1</span></div>
          <HorizonChart models={data.horizons} reliability="p50" selected={teaserModel}
            onSelect={setTeaserModel} reliableRange={data.benchmark.reliable_range_minutes} compact />
        </div>
        <div className="hero-footnote"><span>Measured on selected software tasks. Not “{multiplier}× smarter.”</span>
          <a href="#progress">Read the signal <Arrow /></a></div>
      </section>
      <div className="reading-key">
        <span className="eyebrow">A note before we start</span>
        <div><span className="key-symbol solid" />Measured <small>Published evaluation results</small></div>
        <div><span className="key-symbol hollow" />Estimated <small>Calculated from stated inputs</small></div>
        <div><span className="key-symbol dashed" />Conditional <small>Only if the assumption holds</small></div>
      </div>
      <Progress data={data} />
      <Price data={data} />
      <Scale data={data} />
      <Future data={data} />
      <section className="closing" aria-labelledby="closing-title">
        <div className="eyebrow">The question that started it</div>
        <h2 id="closing-title">And GPT-7?<br /><span>Or 8. Or 9.</span></h2>
        <div className="closing-bottom">
          <p>The names are easy to imagine.<br />The specifications are not ours to invent.</p>
          <p>Better benchmarks, lower prices, longer tasks. Those are signals we can follow.
            What the next model will actually do remains an open question.</p>
        </div>
        <a className="primary-link" href="#future">Explore the assumptions <Arrow /></a>
      </section>
      <section className="sources" id="sources" aria-labelledby="sources-title">
        <div className="source-heading"><div><div className="eyebrow">Behind every dot</div>
          <h2 id="sources-title">Open the notebook.</h2></div>
          <a className="download-link" href={`${import.meta.env.BASE_URL}data/story.json`} download="ai-capability-signals.json">
            Download the data <Arrow down /></a></div>
        <p className="sources-intro">Retrieved {formatDate(data.as_of)}. Each source has its own observation dates.
          A fresh download does not make an old measurement current.</p>
        <div className="source-list">{data.sources.map((source, i) => <div className="source-row" key={source.id}>
          <span className="source-number">0{i + 1}</span>
          <a href={source.page} target="_blank" rel="noreferrer">{source.name}<Arrow /></a>
          <p>{source.description}</p><span className="source-date">{source.retrieved_at.slice(0, 10)}</span>
        </div>)}</div>
        <Evidence title="Methods, coverage, and reproducibility">
          <p>The new dataset contains {data.horizons.length} same-version METR measurements,
            {' '}{data.prices.length} priced and benchmarked catalogue models, and four illustrative
            model-size records. It does not reuse the legacy report’s conclusions.</p>
          <p>All displayed calculations run on a downloaded snapshot. There are no live provider API
            calls in your browser, no tracking, and no user data collection. The downloadable JSON
            includes source URLs, retrieval times, SHA-256 hashes, intervals, exclusion counts, and
            trend checks. Exact raw snapshots are retained locally for hash-verified offline rebuilding.</p>
          <p>METR results describe agent systems, not models in isolation. Artificial Analysis indices
            are passed through OpenRouter, whose response does not expose benchmark version or
            reasoning settings. They are not combined with METR scores. Epoch values are curated
            records, not automatically developer disclosures.</p>
          <p>The data pipeline, tests, and methodology are part of the local rebuild.
            The original v2 implementation remains in <code>archive/v2</code>.
            Source data retains its original terms; code is MIT licensed.</p>
        </Evidence>
      </section>
    </main>
    <footer className="site-footer"><a className="brand" href="#top"><Mark /><span>AI CAPABILITY SIGNALS</span></a>
      <span>A project by Gabriele Monni</span><a href="#top">Back to top ↑</a></footer>
  </>
}

export default function App() {
  const [data, setData] = useState<StoryData | null>(null)
  const [error, setError] = useState(false)
  useEffect(() => {
    const controller = new AbortController()
    fetch(`${import.meta.env.BASE_URL}data/story.json`, { signal: controller.signal })
      .then(response => {
        if (!response.ok) throw new Error('Data unavailable')
        return response.json() as Promise<StoryData>
      }).then(bundle => {
        if (bundle.schema_version !== 1 || !bundle.horizons?.length || !bundle.prices?.length) {
          throw new Error('Invalid story data')
        }
        setData(bundle)
      }).catch(e => { if (e.name !== 'AbortError') setError(true) })
    return () => controller.abort()
  }, [])
  if (error) return <main className="data-state"><Mark /><h1>The data could not be loaded.</h1>
    <p>There are no substitute or invented values. Try loading the local snapshot again.</p>
    <button className="primary-link" onClick={() => window.location.reload()}>Try again <Arrow /></button></main>
  if (!data) return <main className="data-state" aria-busy="true"><Mark /><p>Opening the data story.</p></main>
  return <Story data={data} />
}
