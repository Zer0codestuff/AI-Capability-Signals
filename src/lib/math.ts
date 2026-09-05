import type { Model, Score } from '../types'

export const DAY = 86_400_000
export const time = (date: string) => new Date(`${date}T00:00:00Z`).getTime()

export function duration(minutes: number, compact = false): string {
  if (minutes < 1) return `${Math.round(minutes * 60)}${compact ? 's' : ' sec'}`
  if (minutes < 60) return `${Math.round(minutes)}${compact ? 'm' : ' min'}`
  if (minutes < 1440) {
    const hours = minutes / 60
    return `${hours.toFixed(hours < 10 ? 1 : 0)}${compact ? 'h' : ' hours'}`
  }
  const days = minutes / 1440
  return `${days.toFixed(days < 10 ? 1 : 0)}${compact ? 'd' : ' days'}`
}

export const formatDate = (value: string, short = false) =>
  new Intl.DateTimeFormat('en', {
    month: short ? 'short' : 'long', ...(short ? {} : { day: 'numeric' }),
    year: 'numeric', timeZone: 'UTC',
  }).format(new Date(`${value}T00:00:00Z`))

export function money(value: number) {
  return new Intl.NumberFormat('en-US', {
    style: 'currency', currency: 'USD',
    minimumFractionDigits: value < 1 ? 3 : 2,
    maximumFractionDigits: value < 1 ? 3 : 2,
  }).format(value)
}

export function cost(model: Model, input: number, output: number, requests = 1000): number | null {
  if (input < 0 || output < 0 || requests < 0) return null
  if (input + output > model.context || (model.max_output !== null && output > model.max_output)) return null
  const tier = [...model.tiers].sort((a, b) => b.minimum - a.minimum)
    .find(tier => input >= tier.minimum)
  const prompt = tier?.input ?? model.input
  const completion = tier?.output ?? model.output
  return (prompt * input + completion * output) * requests
}

export type PricedModel = Model & { cost: number; score: number }
export const byScore = (a: PricedModel, b: PricedModel) => b.score - a.score || a.cost - b.cost
export const byName = (a: { name: string }, b: { name: string }) => a.name.localeCompare(b.name)

export function chartDateWindow(dates: string[]) {
  if (!dates.length) throw new Error('A chart needs dates')
  const stamps = dates.map(time)
  const firstYear = new Date(Math.min(...stamps)).getUTCFullYear()
  const last = Math.max(...stamps) + DAY * 60
  const lastYear = new Date(last).getUTCFullYear()
  return {
    domain: [Date.UTC(firstYear, 0, 1), last] as [number, number],
    years: Array.from({ length: lastYear - firstYear + 1 }, (_, i) => firstYear + i),
  }
}

export function pricePoints(models: Model[], metric: Score, input: number, output: number): PricedModel[] {
  return models.flatMap(model => {
    const price = cost(model, input, output)
    const score = model.scores[metric]
    return price !== null && price > 0 && score !== null ? [{ ...model, cost: price, score }] : []
  })
}

export function frontier(points: PricedModel[]): PricedModel[] {
  const ordered = [...points].sort((a, b) => a.cost - b.cost || b.score - a.score)
  let best = -Infinity
  return ordered.filter(point => {
    if (point.score <= best) return false
    best = point.score
    return true
  })
}

export function project(minutes: number, days: number, doublingDays: number, pace: number): number {
  // Python counterpart: pipeline/analysis.py scenario(), with matching known-value tests.
  if (minutes <= 0 || days < 0 || doublingDays <= 0 || pace < 0 || pace > 1.5) {
    throw new Error('Invalid scenario inputs')
  }
  return minutes * 2 ** (days * pace / doublingDays)
}

export function projectCost(current: number, annualDrop: number, years: number): number {
  if (![current, annualDrop, years].every(Number.isFinite) || current < 0 ||
    annualDrop < 0 || annualDrop > 1 || years < 0) throw new Error('Invalid cost scenario')
  return current * (1 - annualDrop) ** years
}

export function linear(domain: [number, number], range: [number, number]) {
  const span = domain[1] - domain[0]
  return (value: number) => range[0] + (span ? (value - domain[0]) / span : 0.5) * (range[1] - range[0])
}

export function logScale(domain: [number, number], range: [number, number]) {
  const scale = linear([Math.log10(domain[0]), Math.log10(domain[1])], range)
  return (value: number) => scale(Math.log10(value))
}
