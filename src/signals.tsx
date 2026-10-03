import type { ReactNode } from 'react'
import type { SignalCopy } from './components/Signal'
import type { Flag, ScaleFacts, Story, Trend } from './types'
import { amount, compact, day, digits, doubling, duration, factor, money, pace, percent, power, span } from './lib/format'

const EPOCH_MODELS = 'https://epoch.ai/data/ai-models'
const EPOCH_BENCHMARKS = 'https://epoch.ai/benchmarks'

const strong = (text: ReactNode) => <strong>{text}</strong>

function years(from: string, to: string) {
  return digits((Date.parse(to) - Date.parse(from)) / (365.25 * 86400e3), 2)
}

function shapeSentence(trend: Trend, noun: string) {
  switch (trend.shape.verdict) {
    case 'speeding_up': return `The ${noun} is rising faster than it used to.`
    case 'slowing_down': return `The ${noun} is rising more slowly than it used to.`
    case 'steady': return `The pace has been steady: the data show no clear slowdown and no clear speed up.`
    default: return `There are too few points to say whether the pace is changing.`
  }
}

/** The largest vetted value, saying plainly whether it is confirmed or an estimate. */
function sizeRecord(facts: ScaleFacts, show: (value: number) => string, noun: string) {
  const top = facts.largest
  const sure = facts.largest_confident
  if (top.c === 'Confident') return `The largest ${noun} on record is ${top.n} (${day(top.d, 'year')}): ${show(top.v)}, confirmed by its maker.`
  return `The largest ${noun} is ${top.n} (${day(top.d, 'year')}): ${show(top.v)}, an outside estimate that Epoch AI rates likely. `
    + `The largest figure confirmed by a maker is ${sure.n} (${day(sure.d, 'year')}): ${show(sure.v)}.`
}

function vettedNote(facts: ScaleFacts, chart: string, flags: Flag[]) {
  const beyond = flags.filter(flag => flag.chart === chart && flag.kind === 'beyond').map(flag => flag.name)
  return `Only values Epoch AI rates confident or likely, for models in its curated set, set records or enter the trend: ${facts.vetted} of ${facts.models}. `
    + 'The rest are grey dots.'
    + (beyond.length ? ` Figures above every vetted value are not drawn at all (${beyond.join(', ')}): they are speculative, or describe what a system could handle rather than a model that was trained.` : '')
}

function fitMethod(trend: Trend, what: string) {
  return <>
    <p>The trend line is an ordinary least squares fit on {what}: {trend.window.n} points
      from {day(trend.window.from, 'month')} to {day(trend.window.to, 'month')}
      {trend.log ? ', fitted to the logarithm of the value so that a straight line means steady multiplication' : ''}.
      The fit explains {percent(trend.r2)} of the variation.</p>
    <p>Ranges come from resampling those points 2,000 times. The shaded projection is the range that should
      contain 80% of new points if the same trend and the same scatter continue. It is a continuation of the
      past, not a forecast of what will happen.</p>
  </>
}

export function signals(story: Story): Record<string, SignalCopy> {
  const flags = story.quality.flags
  const c = story.chapters
  const eci = c.intelligence.charts.eci.series[0].trend!
  const horizon = c.tasks.charts.horizon.series[0].trend!
  const params = c.size.charts.params.series[0].trend!
  const flop = c.compute.charts.compute.series[0].trend!
  const cost = c.cost.charts.cost.series[0].trend!
  const chips = c.hardware.charts.chips.series[0].trend!
  const clusters = c.hardware.charts.clusters.series[0].trend!
  const levels = c.price.facts.levels
  const gpt4 = levels[0]
  const restored = story.quality.launch_prices
  const best = c.price.charts.price.series.find(series => series.id === 'best')?.points ?? []
  const disclosure = c.size.bars.disclosure
  const lastYear = disclosure[disclosure.length - 1]
  const peak = disclosure.reduce((best, row) => (row.params > best.params ? row : best))
  const { intelligence: i, tasks: t, openness: o, race: r } = c
  const leader = r.board.reduce((best, row) => (row.days_on_top > best.days_on_top ? row : best))
  const metrDays = t.facts.published_doubling_days

  return {
    intelligence: {
      id: 'intelligence',
      eyebrow: 'Intelligence',
      question: 'Are models getting smarter?',
      answer: <>Yes. The best model gains about {strong(`${digits(eci.rate.v, 2)} points a year`)} on an index
        that merges dozens of tests into one scale.</>,
      means: [
        <>The Epoch Capabilities Index puts every model on one scale by combining many benchmarks, so a 2026 model
          can be compared with a 2023 one even though the tests changed. Two models anchor the scale: Claude 3.5
          Sonnet is 130 and GPT-5 is 150.</>,
        <>{i.facts.first.n} scored {digits(i.facts.first.v, 3)} in {day(i.facts.first.d, 'month')}.
          {' '}{i.facts.last.n} scored {digits(i.facts.last.v, 3)} in {day(i.facts.last.d, 'month')}:
          {' '}{digits(i.facts.last.v - i.facts.first.v, 2)} points in {years(i.facts.first.d, i.facts.last.d)} years.</>,
        <>{shapeSentence(eci, 'top score')} The index is a score, not a count of anything, so the word exponential
          does not apply here. The next chart measures something that can double.</>,
      ],
      caveats: [
        'A benchmark score is not general intelligence. Models can be tuned to do well on known tests.',
        'The index starts in 2023. In its first ten months nothing beat GPT-4, which is why the early line is flat.',
        `Each lime dot set a new record on release. The other ${i.facts.models - i.facts.records} models are shown in grey.`,
      ],
      method: fitMethod(eci, `the ${i.facts.records} models that set a new record since March 2023`),
      source: { label: 'Epoch AI Benchmarking Hub', href: EPOCH_BENCHMARKS, note: `${i.facts.models} models · latest ${day(i.facts.last.d)}` },
    },

    tasks: {
      id: 'tasks',
      eyebrow: 'Autonomy',
      question: 'How long a task can AI finish on its own?',
      answer: <>About {strong(duration(t.facts.last.v))} of expert work, half the time. That
        length {strong(doubling(horizon) ?? '')}.</>,
      means: [
        <>METR times how long skilled people need for a set of software tasks, then checks which ones a model
          completes alone. The time horizon is the task length a model finishes half the time.</>,
        <>{t.facts.first.n} managed tasks of {duration(t.facts.first.v)} in {day(t.facts.first.d, 'month')}.
          {' '}{t.facts.last.n} reached {duration(t.facts.last.v)} in {day(t.facts.last.d, 'month')}.
          {t.facts.last.p80 ? ` Ask for 80% success instead of 50% and the horizon drops to ${duration(t.facts.last.p80)}.` : ''}</>,
        <>This is the clearest exponential on the page. The scale is logarithmic, so a straight line means the
          length multiplies by {digits(horizon.rate.v, 2)} every year.
          {metrDays ? ` METR's own estimate is a doubling every ${Math.round(metrDays)} days; this fit gives ${Math.round((horizon.doubling?.v ?? 0) * 30.4375)} days.` : ''}</>,
      ],
      caveats: [
        'The tasks are software, machine learning and cybersecurity work. Other kinds of work may behave differently.',
        `METR considers results above ${duration(t.facts.reliable_limit_minutes)} unreliable with the current task set.`,
        `The latest measured model dates from ${day(t.facts.last.d, 'month')}. Newer models have not been measured yet.`,
        'Half the time also means failing half the time. A long horizon is not a reliable worker.',
      ],
      method: <>{fitMethod(horizon, 'the Time Horizon 1.1 models that were state of the art on release since 2023')}
        <p>Three results from the older 1.0 task set are shown in grey but not fitted, because the two task
          sets are not directly comparable.</p></>,
      source: { label: 'METR time horizons', href: 'https://metr.org/time-horizons/', note: `${t.facts.measured} models · latest ${day(t.facts.last.d)}` },
    },

    price: {
      id: 'price-of-use',
      eyebrow: 'Price of use',
      question: 'Is AI getting cheaper to use?',
      answer: <>For a fixed level of ability, dramatically. GPT-4 level answers cost
        {' '}{strong(`${digits(gpt4.fold, 2)} times less`)} than in {day(gpt4.first.d, 'month')}.</>,
      means: [
        <>Prices are per million tokens, roughly 750,000 words, mixing three parts input with one part output.
          Each colored line follows the cheapest model that is at least as capable as the named one.</>,
        <>{gpt4.first.n} cost {money(gpt4.first.v)}. {gpt4.last.n} scores higher on the capability index and
          costs {money(gpt4.last.v)}.
          {levels.slice(1).map(level => ` ${level.label}: ${money(level.first.v)} in ${day(level.first.d, 'month')}, ${money(level.last.v)} now.`).join('')}</>,
        <>The best model of the moment did not get cheaper. Amber dots are the record holders at launch:
          their price ranged from {money(Math.min(...best.map(p => p.v)))} to {money(Math.max(...best.map(p => p.v)))} with
          no downward trend. You pay a premium for the frontier, and yesterday's frontier becomes cheap quickly.</>,
      ],
      caveats: [
        `A price was found for ${c.price.facts.priced} of ${c.price.facts.indexed} models. Retired models with no public price record are missing, so a cheaper option may have existed at some dates.`,
        `${c.price.facts.kinds.observed ?? 0} prices were recorded by Epoch AI. For the others, today's vendor list price stands in for the launch price.`,
        'A hosting price of an open model counts from the day it was recorded, not from the release of the model. Otherwise earlier years would look cheaper than they were.',
        restored.length
          ? `Vendors sometimes cut a price after launch. ${restored.length} known cuts (${restored.map(item => item.model).join(', ')}) are restored to the launch price. An unknown cut would make a step appear too early.`
          : 'Vendors sometimes cut a price after launch. An unknown cut would make a step appear too early.',
        'Some vendors charge more for long prompts. The lowest tier is used.',
        'No projection is drawn: each line has too few steps to test a trend on its own past.',
      ],
      method: <>
        <p>Every model in the capability index is matched to a price in a fixed order of trust: a price Epoch AI
          recorded at the time, then the vendor list price before a documented change, then today's first party
          list price, then the price OpenRouter passes through for closed models.</p>
        <p>For each level, the line is the running minimum price among models released after the reference model
          with an index at least as high. The yearly pace is the total drop spread evenly over the period. No
          trend line is fitted: prices fall in a few large steps, which a straight line describes badly.</p>
      </>,
      caption: 'US dollars per million tokens',
      tiles: levels.map(level => ({
        label: level.label,
        value: `÷${digits(level.fold, 2)}`,
        note: `From ${money(level.first.v)} in ${day(level.first.d, 'month')} to ${money(level.last.v)}. That averages ÷${digits(level.fold ** (1 / Number(years(level.first.d, story.generated_on))), 2)} a year.`,
      })),
      source: { label: 'Epoch AI price trends', href: 'https://epoch.ai/data-insights/llm-inference-price-trends', note: `${c.price.facts.priced} priced models · Epoch AI, llm-prices.com, models.dev, OpenRouter` },
    },

    size: {
      id: 'size',
      eyebrow: 'Model size',
      question: 'Are models getting bigger?',
      answer: <>Yes, but size stopped being the headline. A size is known for
        only {strong(percent(lastYear.params))} of notable models, down from {percent(peak.params)} in {peak.label}.</>,
      means: [
        <>Parameters are the adjustable numbers inside a model, a rough measure of its size. More parameters
          allow more knowledge, and cost more to train and to run.</>,
        <>Among the largest language models, size grew about {factor(params.whole.rate.v)} a year since 2017.
          {' '}{sizeRecord(c.size.facts, v => `${compact(v)} parameters`, 'size')}</>,
        <>Most of the largest models are sparse: they hold trillions of parameters but switch on a small part
          for each input. Size and ability are different things.</>,
      ],
      caveats: [
        params.projectable
          ? 'The projection assumes labs keep publishing sizes, which fewer of them do each year.'
          : 'No projection is drawn. Tested on its own past, this trend did worse than assuming no change.',
        vettedNote(c.size.facts, 'params', flags),
        'Closed labs rarely publish parameter counts. Values for their models are estimates, and say so when you hover them.',
      ],
      method: fitMethod(params, 'vetted language models that ranked among the ten largest at the time of their release, since 2017'),
      source: { label: 'Epoch AI models database', href: EPOCH_MODELS, note: `${c.size.facts.vetted} vetted values among ${c.size.facts.models.toLocaleString('en-US')} language models` },
    },

    compute: {
      id: 'compute',
      eyebrow: 'Training compute',
      question: 'How much computing goes into training?',
      answer: <>{strong(`${digits(flop.rate.v, 2)} times more every year`)} for the largest models since 2010.
        It {doubling(flop)}.</>,
      means: [
        <>Training compute counts the arithmetic operations performed while a model learns. It is the best
          public measure of how much effort went into building it.</>,
        <>{sizeRecord(c.compute.facts, v => `about ${power(v)} operations`, 'training run')}
          {' '}{shapeSentence(flop, 'amount of compute')}</>,
        <>A steady multiplication for sixteen years is what exponential growth looks like. It is also far
          faster than chips improve, so most of it comes from spending more and building bigger clusters.</>,
      ],
      caveats: [
        'Most recent values are estimates by Epoch AI researchers, not company disclosures.',
        vettedNote(c.compute.facts, 'compute', flags),
        `Only ${percent(lastYear.compute)} of notable ${lastYear.label} language models have a compute estimate, so the recent frontier is thinly covered.`,
        'More compute does not guarantee a better model. It measures input, not result.',
      ],
      method: fitMethod(flop, 'vetted models that ranked among the ten largest training runs at the time of release, since 2010'),
      source: { label: 'Epoch AI models database', href: EPOCH_MODELS, note: `${c.compute.facts.vetted} vetted values among ${c.compute.facts.models.toLocaleString('en-US')} models` },
    },

    cost: {
      id: 'training-cost',
      eyebrow: 'Training cost',
      question: 'What does it cost to train a top model?',
      answer: <>About {strong(`${digits(cost.rate.v, 2)} times more every year`)}. The most expensive run with a
        vetted estimate cost {strong(amount('usd', c.cost.facts.largest.v))}.</>,
      means: [
        <>This is the price of the computing used in the final training run, in 2023 dollars. It leaves out
          salaries, failed experiments, data and everything else a lab spends.</>,
        <>{sizeRecord(c.cost.facts, v => amount('usd', v), 'training run')} Cost rises more slowly
          than compute because each dollar buys more computing every year.</>,
        <>{shapeSentence(cost, 'cost')} At this pace the bill {doubling(cost)}.</>,
      ],
      caveats: [
        `Estimates are sparse for recent models: ${c.cost.facts.estimates_last_two_years} vetted ones in the last two years. The latest is from ${day(cost.window.to, 'month')}.`,
        vettedNote(c.cost.facts, 'cost', flags),
        'Every value is an estimate built from hardware, duration and cloud prices. None is an audited figure.',
        'The full cost of developing a model is several times the cost of its final run.',
      ],
      method: fitMethod(cost, 'vetted models that ranked among the ten most expensive at the time of release, since 2016'),
      source: { label: 'Epoch AI models database', href: EPOCH_MODELS, note: `${c.cost.facts.vetted} vetted values among ${c.cost.facts.models} cost estimates` },
    },

    chips: {
      id: 'chips',
      eyebrow: 'Hardware value',
      question: 'Do chips give more for the money?',
      answer: <>Yes, about {strong(`${digits(chips.rate.v, 2)} times more computing per dollar`)} each year.
        Value for money {doubling(chips)}.</>,
      means: [
        <>Each dot is a chip at its launch price. The value is how many operations per second one dollar of
          hardware performs, at the 32 or 16 bit precision used to train models.</>,
        <>{shapeSentence(chips, 'value for money')} Compare this with training compute, which
          grows {digits(flop.rate.v, 2)} times a year: better chips explain only a small part of it.</>,
        <>Gaming cards hold the records because they are cheap. Data center chips cost more per operation,
          but pack far more computing and memory into one machine.</>,
      ],
      caveats: [
        'Every chip is measured the same way. The 8 and 4 bit formats that newer chips add for running models are left out, because counting them would make progress look faster than it is.',
        `Only ${c.hardware.facts.chips} chips have a public launch price. Many data center chips are sold at negotiated prices.`,
        'Launch price ignores electricity, cooling and networking, which are a large share of real cost.',
      ],
      method: fitMethod(chips, 'every chip with a known launch price since 2012, using the fastest of its 32 and 16 bit speeds'),
      source: { label: 'Epoch AI hardware database', href: 'https://epoch.ai/data/machine-learning-hardware', note: `${c.hardware.facts.chips} chips with a launch price` },
    },

    clusters: {
      id: 'clusters',
      eyebrow: 'Data centers',
      question: 'How big are the machines that train AI?',
      answer: <>The largest AI clusters grow about {strong(`${digits(clusters.rate.v, 2)} times a year`)}.
        The biggest one recorded equals {strong(compact(c.hardware.facts.largest.v))} top Nvidia H100 chips.</>,
      means: [
        <>A cluster is one site full of AI chips working together. Size is expressed in H100 equivalents so
          that sites with different chips can be compared.</>,
        <>The largest recorded site is {c.hardware.facts.largest.n}, switched on in {day(c.hardware.facts.largest.d, 'month')}.
          {' '}{shapeSentence(clusters, 'size of the largest clusters')}</>,
      ],
      caveats: [
        `The database stops at ${day(c.hardware.facts.clusters_through, 'month')}. Sites switched on since then are missing.`,
        'Many sizes are estimated from chip orders, power capacity or satellite images.',
      ],
      method: fitMethod(clusters, 'clusters that ranked among the ten largest when they were switched on, since 2019'),
      source: { label: 'Epoch AI supercomputers database', href: 'https://epoch.ai/data/ai-supercomputers', note: `${c.hardware.facts.clusters} clusters · through ${day(c.hardware.facts.clusters_through, 'month')}` },
    },

    openness: {
      id: 'open-models',
      eyebrow: 'Open and closed',
      question: 'How far behind are open models?',
      answer: <>About {strong(span(o.facts.lag.now))}. The best open model today matches what closed models could
        do when {o.facts.lag.matched} came out.</>,
      means: [
        <>Open models can be downloaded and run by anyone. Closed models are reachable only through their
          maker's service. The gap between the two lines is how much capability is kept behind that door.</>,
        <>The best open model is {o.facts.open.n} at {digits(o.facts.open.v, 4)}. The best closed model
          is {o.facts.closed.n} at {digits(o.facts.closed.v, 4)}.</>,
        o.facts.lag.recent_average !== null && o.facts.lag.earlier_average !== null
          ? <>In the last two years each open record arrived on average {span(o.facts.lag.recent_average)} after
            closed models reached the same level. Before that the average was {span(o.facts.lag.earlier_average)},
            mostly because it took open models over a year to match GPT-4.</>
          : null,
      ].filter(Boolean),
      caveats: [
        `${o.facts.counts.other} models with unclear access terms are shown in grey and belong to neither line.`,
        'The lag measured today keeps growing until the next open record, so it reads high between releases.',
        'Open weights do not mean cheap to run. The largest open models need data center hardware.',
      ],
      tiles: [
        { label: 'Gap today', value: span(o.facts.lag.now), note: `The best open model matches the level closed models reached with ${o.facts.lag.matched}.` },
        { label: 'Gap in score', value: `${digits(o.facts.closed.v - o.facts.open.v, 2)} points`, note: `${o.facts.closed.n} at ${digits(o.facts.closed.v, 4)} against ${o.facts.open.n} at ${digits(o.facts.open.v, 4)}.` },
        { label: 'Typical delay, last two years', value: o.facts.lag.recent_average === null ? 'Unknown' : span(o.facts.lag.recent_average),
          note: o.facts.lag.earlier_average === null ? '' : `Between a closed record and the open model that matched it. Before that: ${span(o.facts.lag.earlier_average)}.` },
      ],
      method: <p>Each line is the running best index score among models of that kind. For every open record the
        lag is the time since a closed model first reached that score.</p>,
      source: { label: 'Epoch AI Benchmarking Hub', href: EPOCH_BENCHMARKS, note: `${o.facts.counts.open} open and ${o.facts.counts.closed} closed models` },
    },

    race: {
      id: 'countries',
      eyebrow: 'Countries and labs',
      question: 'Who is ahead?',
      answer: <>US labs hold the top spot. The best Chinese model trails by about {strong(span(r.facts.lag.now))}.</>,
      means: [
        <>The best US model is {r.facts.us.n} at {digits(r.facts.us.v, 4)}. The best Chinese model
          is {r.facts.china.n} at {digits(r.facts.china.v, 4)}, a level US models reached with {r.facts.lag.matched}.</>,
        <>{leader.org} held the overall record for {leader.days_on_top.toLocaleString('en-US')} of
          the {r.facts.days_tracked.toLocaleString('en-US')} days tracked. The table below lists the best model
          of each lab.</>,
        <>Most leading Chinese models are open, so this gap and the open model gap are nearly the same story.</>,
      ],
      caveats: [
        'A model is assigned to the country of the organization that made it.',
        'The index covers models Epoch AI could evaluate. Models never released outside a lab are not counted.',
      ],
      tiles: [
        { label: 'Gap today', value: span(r.facts.lag.now), note: `The best Chinese model matches the level US models reached with ${r.facts.lag.matched}.` },
        { label: 'Gap in score', value: `${digits(r.facts.us.v - r.facts.china.v, 2)} points`, note: `${r.facts.us.n} at ${digits(r.facts.us.v, 4)} against ${r.facts.china.n} at ${digits(r.facts.china.v, 4)}.` },
        { label: 'Longest at the top', value: leader.org, note: `${leader.days_on_top.toLocaleString('en-US')} of ${r.facts.days_tracked.toLocaleString('en-US')} days tracked, with ${leader.records} record setting models.` },
      ],
      method: <p>Each line is the running best index score among models from that country. The lag is the
        time since a US model first reached the score of each Chinese record.</p>,
      source: { label: 'Epoch AI Benchmarking Hub', href: EPOCH_BENCHMARKS, note: `${r.facts.counts.us} US, ${r.facts.counts.china} Chinese and ${r.facts.counts.other} other models` },
    },
  }
}

export interface Headline { href: string; label: string; value: string; note: string }

export function headlines(story: Story): Headline[] {
  const c = story.chapters
  const eci = c.intelligence.charts.eci.series[0].trend!
  const horizon = c.tasks.charts.horizon.series[0].trend!
  const flop = c.compute.charts.compute.series[0].trend!
  const cost = c.cost.charts.cost.series[0].trend!
  const gpt4 = c.price.facts.levels[0]
  return [
    { href: '#intelligence', label: 'Capability index', value: pace(eci, false).replace(' a year', ''), note: 'gained by the best model each year' },
    { href: '#tasks', label: 'Task length', value: `×2 every ${span(horizon.doubling?.v ?? 0)}`, note: 'for work a model can finish alone' },
    { href: '#price-of-use', label: 'Price of GPT-4 level', value: `÷${digits(gpt4.fold, 2)}`, note: `since ${day(gpt4.first.d, 'month')}` },
    { href: '#compute', label: 'Training compute', value: `${factor(flop.rate.v)} a year`, note: 'for the largest models since 2010' },
    { href: '#training-cost', label: 'Training cost', value: `${factor(cost.rate.v)} a year`, note: 'for the most expensive runs' },
    { href: '#open-models', label: 'Open models', value: span(c.openness.facts.lag.now), note: 'behind the best closed model' },
  ]
}
