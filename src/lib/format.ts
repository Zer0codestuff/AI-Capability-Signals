import type { Backtest, Interval, Trend, Unit } from '../types'

const SUPERSCRIPT = '⁰¹²³⁴⁵⁶⁷⁸⁹'
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

export const time = (iso: string) => Date.parse(`${iso}T00:00:00Z`)

export function day(iso: string, style: 'full' | 'month' | 'year' = 'full') {
  const [year, month, date] = iso.split('-').map(Number)
  if (style === 'year') return String(year)
  if (style === 'month') return `${MONTHS[month - 1]} ${year}`
  return `${date} ${MONTHS[month - 1]} ${year}`
}

/** Trim to a sensible number of digits without trailing zeros. */
export function digits(value: number, significant = 3) {
  if (value === 0) return '0'
  const places = Math.max(0, significant - 1 - Math.floor(Math.log10(Math.abs(value))))
  return value.toLocaleString('en-US', { maximumFractionDigits: Math.min(places, 6) })
}

export function compact(value: number, significant = 3) {
  const steps: [number, string][] = [[1e15, 'Q'], [1e12, 'T'], [1e9, 'B'], [1e6, 'M'], [1e3, 'K']]
  for (const [size, suffix] of steps) if (Math.abs(value) >= size) return digits(value / size, significant) + suffix
  return digits(value, significant)
}

export function power(value: number, withMantissa = true) {
  const exponent = Math.floor(Math.log10(value) + 1e-9)
  const mantissa = value / 10 ** exponent
  const raised = '10' + String(exponent).split('').map(c => (c === '-' ? '⁻' : SUPERSCRIPT[+c])).join('')
  return withMantissa && Math.abs(mantissa - 1) > 0.05 ? `${digits(mantissa, 2)} × ${raised}` : raised
}

export function duration(minutes: number) {
  if (minutes < 1) return `${digits(minutes * 60, 2)} sec`
  if (minutes < 60) return `${digits(minutes, 2)} min`
  const hours = minutes / 60
  return `${hours < 100 ? digits(hours, 2) : Math.round(hours).toLocaleString('en-US')} h`
}

export function money(value: number) {
  if (value >= 1000) return `$${compact(value)}`
  if (value >= 1) return `$${value.toFixed(2)}`
  return `$${digits(value, 2)}`
}

/** A value with its unit, short enough for an axis tick or a tooltip. */
export function amount(unit: Unit, value: number, tick = false): string {
  switch (unit) {
    case 'eci': return tick ? String(Math.round(value)) : value.toFixed(1)
    case 'minutes': return duration(value)
    case 'params': return compact(value, tick ? 2 : 3)
    case 'flop': return power(value, !tick)
    case 'usd': return `$${compact(value, tick ? 2 : 3)}`
    case 'usd_mtok': return tick ? `$${value >= 1 ? digits(value, 3) : value}` : money(value)
    case 'ops_per_usd': return compact(value, tick ? 2 : 3)
    case 'h100e': return compact(value, tick ? 2 : 3)
    case 'months': return tick ? String(Math.round(value)) : `${digits(value, 2)} months`
  }
}

/** Axis marks for task length, in minutes: seconds, minutes, hours, then longer. */
export const TIME_TICKS = [1 / 60, 1 / 6, 1, 10, 60, 600, 6000, 60000, 600000, 6000000]

export const UNIT_LABEL: Record<Unit, string> = {
  eci: 'Epoch Capabilities Index',
  minutes: 'Task length a model completes half the time',
  params: 'Parameters',
  flop: 'Training compute, in operations (FLOP)',
  usd: 'Training cost, 2023 US dollars',
  usd_mtok: 'Price per million tokens',
  ops_per_usd: 'Operations per second for each dollar',
  h100e: 'Cluster size, in Nvidia H100 equivalents',
  months: 'Months behind',
}

/** "×4.6" for growth, "÷8.7" for decline. */
export function factor(value: number) {
  return value >= 1 ? `×${digits(value, value >= 10 ? 2 : 2)}` : `÷${digits(1 / value, 2)}`
}

export function span(months: number) {
  const size = Math.abs(months)
  if (size >= 24) return `${digits(size / 12, 2)} years`
  return `${digits(size, 2)} months`
}

export function pace(trend: { rate: Interval; doubling?: Interval }, log: boolean) {
  if (!log) return `${trend.rate.v >= 0 ? '+' : ''}${digits(trend.rate.v, 3)} points a year`
  return `${factor(trend.rate.v)} a year`
}

export function paceRange(trend: Trend) {
  const { lo, hi } = trend.rate
  if (lo === undefined || hi === undefined) return ''
  if (!trend.log) return `${digits(lo, 3)} to ${digits(hi, 3)} points a year`
  const ends = trend.rate.v >= 1 ? [lo, hi] : [hi, lo]
  return `${factor(ends[0])} to ${factor(ends[1])} a year`
}

export function doubling(trend: { doubling?: Interval }) {
  if (!trend.doubling) return null
  const months = trend.doubling.v
  return `${months > 0 ? 'doubles' : 'halves'} every ${span(months)}`
}

/** How far off the method was, in words a reader can weigh. */
export function miss(check: Backtest, value: number) {
  return check.log ? `a factor of ${digits(10 ** value, 2)}` : `${digits(value, 2)} points`
}

export function percent(share: number) {
  return `${Math.round(share * 100)}%`
}
