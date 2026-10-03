import { useEffect, useMemo, useState } from 'react'
import type { Story } from './types'
import Nav, { SECTIONS } from './components/Nav'
import Signal from './components/Signal'
import Explorer from './components/Explorer'
import Method from './components/Method'
import { Board, DisclosureBars, LagCard } from './components/Extras'
import { Reveal, ArrowDown, Mark } from './components/ui'
import { headlines, signals } from './signals'
import { day } from './lib/format'

function useStory() {
  const [story, setStory] = useState<Story | null>(null)
  const [failed, setFailed] = useState(false)
  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/signals.json`)
      .then(response => (response.ok ? response.json() : Promise.reject(response.status)))
      .then(setStory)
      .catch(() => setFailed(true))
  }, [])
  return { story, failed }
}

function useActiveSection(ready: boolean) {
  const [active, setActive] = useState('')
  useEffect(() => {
    if (!ready || typeof IntersectionObserver === 'undefined') return
    const observer = new IntersectionObserver(entries => {
      for (const entry of entries) if (entry.isIntersecting) setActive(entry.target.id)
    }, { rootMargin: '-40% 0px -55% 0px' })
    for (const section of SECTIONS) {
      const node = document.getElementById(section.id)
      if (node) observer.observe(node)
    }
    return () => observer.disconnect()
  }, [ready])
  return active
}

function Part({ id, title, children }: { id: string; title: string; children: React.ReactNode }) {
  return <div className="part" id={id}>
    <Reveal className="part-title"><span>{title}</span></Reveal>
    {children}
  </div>
}

export default function App() {
  const { story, failed } = useStory()
  const active = useActiveSection(!!story)
  const copy = useMemo(() => (story ? signals(story) : null), [story])

  // The data arrives after the first paint, so a direct link to a chart needs a nudge.
  useEffect(() => {
    if (!story || !location.hash) return
    document.getElementById(location.hash.slice(1))?.scrollIntoView()
  }, [story])

  if (failed) return <main className="state"><h1>The data could not be loaded.</h1><p>Reload the page to try again.</p></main>
  if (!story || !copy) return <main className="state" aria-busy="true" />

  const c = story.chapters
  const today = story.generated_on

  return <>
    <a className="skip-link" href="#capability">Skip to the charts</a>
    <Nav active={active} />
    <main id="top">
      <header className="hero">
        <p className="eyebrow"><span className="dot" />Public data, updated {day(today)}</p>
        <h1>How fast is AI moving?</h1>
        <p className="lede">Ten measured trends about what AI models can do, what they cost and what it takes
          to build them. Plain answers first, the evidence right below.</p>
        <div className="headlines">
          {headlines(story).map(item => <a className="card headline" href={item.href} key={item.href}>
            <span className="stat-label">{item.label}</span>
            <strong>{item.value}</strong>
            <span className="headline-note">{item.note}</span>
          </a>)}
        </div>
        <div className="reading-key">
          <span><i className="key-dot" style={{ background: 'var(--dot)' }} />Each dot is a model</span>
          <span><i className="key-dot" style={{ background: 'var(--accent)' }} />Color marks the leaders</span>
          <span><i className="key-line fit-key" />The line is the trend</span>
          <span><i className="key-band" />The shade is where it may go</span>
          <a href="#capability">Start reading<ArrowDown size={14} aria-hidden="true" /></a>
        </div>
      </header>

      <Part id="capability" title="What models can do">
        <Signal copy={copy.intelligence} number="01" chart={c.intelligence.charts.eci} today={today} />
        <Signal copy={copy.tasks} number="02" chart={c.tasks.charts.horizon} today={today} />
      </Part>

      <Part id="price" title="What it costs to use them">
        <Signal copy={copy.price} number="03" chart={c.price.charts.price} today={today} />
      </Part>

      <Part id="scale" title="What it takes to build them">
        <Signal copy={copy.size} number="04" chart={c.size.charts.params} today={today}>
          <DisclosureBars rows={c.size.bars.disclosure} />
        </Signal>
        <Signal copy={copy.compute} number="05" chart={c.compute.charts.compute} today={today} />
        <Signal copy={copy.cost} number="06" chart={c.cost.charts.cost} today={today} />
        <Signal copy={copy.chips} number="07" chart={c.hardware.charts.chips} today={today} />
        <Signal copy={copy.clusters} number="08" chart={c.hardware.charts.clusters} today={today} />
      </Part>

      <Part id="race" title="Who is ahead">
        <Signal copy={copy.openness} number="09" chart={c.openness.charts.access} today={today}>
          <LagCard chart={c.openness.charts.access_lag} today={today}
            title="Months each open record arrived after closed models reached the same level" />
        </Signal>
        <Signal copy={copy.race} number="10" chart={c.race.charts.country} today={today}>
          <Board rows={c.race.board} tracked={c.race.facts.days_tracked} />
        </Signal>
      </Part>

      <Explorer rows={story.explorer} />
      <Method story={story} />
    </main>
    <footer className="footer">
      <span className="brand"><Mark />AI Capability Signals</span>
      <span>Data through {day(story.data_through)}. Code under MIT, data under the terms of each source.</span>
    </footer>
  </>
}
