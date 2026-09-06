import type { StoryData } from '../types'
import { duration, formatDate } from '../lib/math'
import { Plus, ArrowUpRight } from '@phosphor-icons/react/dist/ssr'
import { Reveal } from './ui'

export function faqItems(data: StoryData) {
  const first = data.horizons.find(m => m.id === 'gpt_4') ?? data.horizons[0]
  const anchor = data.horizons.find(m => m.id === data.trends.p50.anchor_id)!
  const latest = data.horizons.at(-1)!
  const multiplier = Math.round(anchor.p50.estimate / first.p50.estimate)
  const limit = duration(data.benchmark.reliable_range_minutes, true)
  return [
    {
      q: 'Is this a prediction about GPT-7, GPT-8 or GPT-9?',
      a: `No. The future chapter extends a measured trend under three explicit assumptions: progress stops, slows to half pace, or continues at the fitted pace. No probability is attached to any path, and no specification, release date or score is invented for an unreleased model.`,
    },
    {
      q: `Does a ${multiplier}× longer task horizon mean ${multiplier}× smarter?`,
      a: `No. The horizon is the length of a software task, measured in the time a human expert would need, that an agent completes at a given success rate. ${first.name} sits at ${duration(first.p50.estimate)} and ${anchor.name} at ${duration(anchor.p50.estimate)} at 50% success. It is one benchmark suite on software, machine learning and cybersecurity tasks, not a general intelligence score.`,
    },
    {
      q: 'What does "50% success" mean, and why does the chart shrink at 80%?',
      a: 'At 50% success, half of the comparable benchmark tasks of that length succeed. At 80%, about eight in ten do. Reliability costs time: every model handles much shorter tasks when you demand a higher success rate, so the 80% view is the more conservative reading.',
    },
    {
      q: `Why is everything above ${limit} shaded?`,
      a: `METR states that its current task suite cannot measure horizons above ${limit} reliably. Points and scenario paths in that region are shown so you can see them, but they should not be read as precise results.`,
    },
    {
      q: 'Why does the measured series stop in ' + formatDate(latest.release_date, true) + '?',
      a: `Only Time Horizon 1.1 results are included, and ${formatDate(latest.release_date)} is the latest model release with a published same version measurement at the retrieval date. Three older TH 1.0 records embedded in the source are excluded rather than mixed in. A newer retrieval date does not make an old measurement current.`,
    },
    {
      q: 'Are the prices historical, and is the cost chart a capability ranking?',
      a: 'No on both counts. Prices are current list prices from the OpenRouter catalogue on the retrieval date, computed for a declared token budget. They exclude caching, tool fees, retries and extra reasoning tokens. The benchmark indices come from Artificial Analysis and their version and reasoning settings are not exposed, so the comparison is indicative, not a controlled series.',
    },
    {
      q: 'Why is there no parameter forecast?',
      a: 'Parameter counts describe model size, not intelligence, and several current models do not disclose them at all. A populated field in a database can be a researcher estimate rather than a developer disclosure. Extrapolating a size curve would say nothing reliable about what a future model can do.',
    },
    {
      q: 'Can I check the numbers myself?',
      a: `Yes. The downloadable JSON holds every value on this page with source URLs, retrieval times, SHA-256 hashes, confidence intervals, exclusion counts and the historical trend check. The site makes no live provider calls and collects no user data. Data was retrieved ${formatDate(data.as_of)}.`,
    },
  ]
}

export default function Faq({ data }: { data: StoryData }) {
  const items = faqItems(data)
  const schema = {
    '@context': 'https://schema.org', '@type': 'FAQPage',
    mainEntity: items.map(item => ({
      '@type': 'Question', name: item.q,
      acceptedAnswer: { '@type': 'Answer', text: item.a },
    })),
  }
  return <section id="faq" className="chapter faq" aria-labelledby="faq-title">
    <Reveal as="header" className="section-head">
      <p className="eyebrow"><span className="chapter-number" aria-hidden="true">05</span>Questions</p>
      <div className="section-intro"><h2 id="faq-title">Read this before<br />you quote a number.</h2>
        <p>Eight questions that come up every time the charts are shared. Each answer is
          short, and each one is backed by the sources at the end.</p></div>
    </Reveal>
    <Reveal className="faq-list">
      {items.map((item, i) => <details className="faq-item" key={item.q}>
        <summary><span className="faq-number" aria-hidden="true">{String(i + 1).padStart(2, '0')}</span>
          <span className="faq-question">{item.q}</span>
          <Plus size={20} aria-hidden="true" /></summary>
        <p>{item.a}</p>
      </details>)}
    </Reveal>
    <Reveal className="faq-foot">
      <a href="https://metr.org/notes/2026-01-22-time-horizon-limitations/" target="_blank" rel="noreferrer">
        METR on the limits of time horizons<ArrowUpRight size={14} aria-hidden="true" /></a>
      <a href="https://artificialanalysis.ai/methodology/intelligence-benchmarking" target="_blank" rel="noreferrer">
        Artificial Analysis benchmark methodology<ArrowUpRight size={14} aria-hidden="true" /></a>
    </Reveal>
    <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(schema) }} />
  </section>
}
