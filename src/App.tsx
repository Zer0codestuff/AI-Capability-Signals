import { useEffect, useState } from 'react'
import type { StoryData } from './types'
import { duration, formatDate } from './lib/math'
import { ArrowDown, ArrowRight, ArrowUp, ArrowUpRight, DownloadSimple, Evidence, Mark, Reveal } from './components/ui'
import Nav from './components/Nav'
import HorizonChart from './components/HorizonChart'
import Progress from './components/Progress'
import Price from './components/Price'
import Scale from './components/Scale'
import Future from './components/Future'
import Faq from './components/Faq'
import Tagline from './components/Tagline'

const ANCHORS = ['top', 'progress', 'cost', 'scale', 'future', 'faq', 'sources', 'privacy', 'terms']

function useActiveChapter() {
  const [active, setActive] = useState('')
  useEffect(() => {
    const observer = new IntersectionObserver(entries => {
      const visible = entries.filter(entry => entry.isIntersecting)
      if (visible.length) setActive(visible[0].target.id)
    }, { rootMargin: '-20% 0px -60% 0px' })
    document.querySelectorAll('main > section[id]').forEach(section => observer.observe(section))
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
  const dataHref = `${import.meta.env.BASE_URL}data/story.json`
  useEffect(() => {
    const id = window.location.hash.slice(1)
    if (!ANCHORS.includes(id)) return
    const frame = requestAnimationFrame(() =>
      document.getElementById(id)?.scrollIntoView({ behavior: 'instant' }))
    return () => cancelAnimationFrame(frame)
  }, [])
  return <>
    <a className="skip-link" href="#main">Skip to the story</a>
    <Nav active={active} />
    <main id="main">
      <section className="hero" id="top">
        <Reveal className="hero-overline">
          <span className="eyebrow"><span className="dot" />An independent data story</span>
          <span className="update-date">Data retrieved {formatDate(data.as_of, true)}</span>
        </Reveal>
        <div className="hero-grid">
          <Reveal className="hero-copy" delay={80}>
            <h1>AI is moving fast.<br />How far can it go?</h1>
            <p>Measured progress, current prices and conditional futures for the next
              generation of AI, with the uncertainty left visible.</p>
            <a className="button primary" href="#progress">Follow the evidence<ArrowDown size={18} aria-hidden="true" /></a>
          </Reveal>
          <Reveal as="div" className="card hero-proof" delay={160}>
            <span className="stat-label">One measured change</span>
            <strong><span className="num">{multiplier}</span><span className="times">×</span></strong>
            <span className="hero-proof-label">longer task horizon</span>
            <p>{duration(first.p50.estimate)} <ArrowRight size={12} aria-hidden="true" /> {duration(anchor.p50.estimate)}
              <br />{first.release_date.slice(0, 4)} to {anchor.release_date.slice(0, 4)} · 50% success · METR</p>
          </Reveal>
        </div>
        <Reveal className="card hero-chart" delay={240}>
          <div className="chart-caption"><span className="eyebrow">Human task time</span>
            <span aria-live="polite" className="mono">{teaser.name} · {duration(teaser.p50.estimate)} · METR TH 1.1</span></div>
          <HorizonChart models={data.horizons} reliability="p50" selected={teaserModel}
            onSelect={setTeaserModel} reliableRange={data.benchmark.reliable_range_minutes} compact />
          <div className="chart-foot"><span>Measured on selected software tasks. Not “{multiplier}× smarter”.</span>
            <a href="#progress">Read the signal<ArrowRight size={14} aria-hidden="true" /></a></div>
        </Reveal>
      </section>

      <Reveal className="reading-key" as="section">
        <span className="eyebrow">How to read the marks</span>
        <div><span className="key-symbol solid" /><span>Measured</span><small>Published evaluation results</small></div>
        <div><span className="key-symbol hollow" /><span>Estimated</span><small>Calculated from stated inputs</small></div>
        <div><span className="key-symbol dashed" /><span>Conditional</span><small>Only if the assumption holds</small></div>
      </Reveal>

      <Progress data={data} />
      <Price data={data} />
      <Scale data={data} />

      <Tagline label="What the data can and cannot tell you" lines={[
        'Better benchmarks, lower prices, longer tasks.',
        'Those are signals you can follow.',
        'What the next model will do is not one of them.',
      ]} />

      <Future data={data} />
      <Faq data={data} />

      <section className="closing" id="closing" aria-labelledby="closing-title">
        <Reveal>
          <p className="eyebrow">The question that started it</p>
          <h2 id="closing-title">And GPT-7?<br /><span>Or 8. Or 9.</span></h2>
        </Reveal>
        <Reveal className="closing-grid" delay={100}>
          <p>The names are easy to imagine.<br />The specifications are not ours to invent.</p>
          <div className="closing-actions">
            <a className="button primary" href={dataHref} download="ai-capability-signals.json">
              Download the data<DownloadSimple size={18} aria-hidden="true" /></a>
            <a className="button ghost" href="#future">Revisit the assumptions<ArrowUp size={18} aria-hidden="true" /></a>
          </div>
        </Reveal>
      </section>

      <section className="sources" id="sources" aria-labelledby="sources-title">
        <Reveal as="header" className="source-heading">
          <div><p className="eyebrow">Behind every dot</p><h2 id="sources-title">Open the notebook.</h2></div>
          <p>Retrieved {formatDate(data.as_of)}. Each source has its own observation dates.
            A fresh download does not make an old measurement current.</p>
        </Reveal>
        <Reveal className="source-list" delay={80}>
          {data.sources.map((source, i) => <a className="card source-row" key={source.id}
            href={source.page} target="_blank" rel="noreferrer">
            <span className="source-number">0{i + 1}</span>
            <span className="source-name">{source.name}<ArrowUpRight size={16} aria-hidden="true" /></span>
            <p>{source.description}</p>
            <span className="source-date mono">{source.retrieved_at.slice(0, 10)}</span>
          </a>)}
        </Reveal>
        <Reveal delay={120}>
          <Evidence title="Methods, coverage, and reproducibility">
            <p>The dataset contains {data.horizons.length} same version METR measurements,
              {' '}{data.prices.length} priced and benchmarked catalogue models, and four illustrative
              model size records. It does not reuse the legacy report’s conclusions.</p>
            <p>All displayed calculations run on a downloaded snapshot. There are no live provider API
              calls in your browser, no tracking, and no user data collection. The downloadable JSON
              includes source URLs, retrieval times, SHA-256 hashes, intervals, exclusion counts, and
              trend checks. Exact raw snapshots are retained locally for hash verified offline rebuilding.</p>
            <p>METR results describe agent systems, not models in isolation. Artificial Analysis indices
              are passed through OpenRouter, whose response does not expose benchmark version or
              reasoning settings. They are not combined with METR scores. Epoch values are curated
              records, not automatically developer disclosures.</p>
            <p>The data pipeline, tests, and methodology are part of the local rebuild.
              The original v2 implementation remains in <code>archive/v2</code>.
              Source data retains its original terms; code is MIT licensed.</p>
          </Evidence>
          <div className="legal">
            <div id="privacy"><h3>Privacy</h3>
              <p>No cookies, no analytics, no accounts. The page loads one JSON file from its own
                origin and nothing else. Nothing you select here leaves your browser.</p></div>
            <div id="terms"><h3>Terms</h3>
              <p>The code is MIT licensed. METR, OpenRouter, Artificial Analysis and Epoch AI data keep
                their original terms; review them before commercial or bulk redistribution.</p></div>
          </div>
        </Reveal>
      </section>
    </main>
    <footer className="site-footer">
      <a className="brand" href="#top"><Mark /><span>AI Capability Signals</span></a>
      <span className="footer-note">A project by Gabriele Monni</span>
      <nav aria-label="Legal"><a href="#privacy">Privacy</a><a href="#terms">Terms</a>
        <a href={dataHref} download="ai-capability-signals.json">Data</a></nav>
      <a className="footer-top" href="#top">Back to top<ArrowUp size={14} aria-hidden="true" /></a>
    </footer>
  </>
}

function Skeleton() {
  return <main className="data-state" aria-busy="true">
    <p className="visually-hidden">Opening the data story.</p>
    <div className="skeleton skeleton-pill" />
    <div className="skeleton skeleton-title" />
    <div className="skeleton skeleton-title short" />
    <div className="skeleton skeleton-line" />
    <div className="skeleton skeleton-chart" />
  </main>
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
  if (error) return <main className="data-state" role="alert"><Mark />
    <h1>The data could not be loaded.</h1>
    <p>There are no substitute or invented values. Reload to try the local snapshot again.</p>
    <button className="button primary" onClick={() => window.location.reload()}>Try again<ArrowRight size={18} aria-hidden="true" /></button></main>
  if (!data) return <Skeleton />
  return <Story data={data} />
}
