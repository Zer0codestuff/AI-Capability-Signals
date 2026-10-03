import { describe, expect, it } from 'vitest'
import { amount, compact, day, duration, factor, money, power, span } from './format'
import { linearScale, logScale, tickedLogScale, yearTicks } from './scale'

describe('formatting', () => {
  it('writes quantities the way a reader says them', () => {
    expect(duration(0.5)).toBe('30 sec')
    expect(duration(45)).toBe('45 min')
    expect(duration(1045)).toBe('17 h')
    expect(duration(120000)).toBe('2,000 h')
    expect(compact(2.8e12)).toBe('2.8T')
    expect(money(0.055)).toBe('$0.055')
    expect(money(37.5)).toBe('$37.50')
    expect(power(1e27)).toBe('10²⁷')
    expect(power(3.14e23)).toBe('3.1 × 10²³')
    expect(factor(4.62)).toBe('×4.6')
    expect(factor(0.1145)).toBe('÷8.7')
    expect(span(4.1)).toBe('4.1 months')
    expect(span(-30)).toBe('2.5 years')
    expect(day('2026-09-22')).toBe('22 Sep 2026')
    expect(amount('usd_mtok', 0.1, true)).toBe('$0.1')
  })
})

describe('scales', () => {
  it('maps the domain ends onto the plot ends', () => {
    const linear = linearScale(103, 167, 400, 0)
    expect(linear(linear.domain[0])).toBe(400)
    expect(linear(linear.domain[1])).toBe(0)
    const log = logScale(3e16, 1e27, 400, 0)
    expect(log(log.domain[0])).toBeCloseTo(400)
    expect(log(log.domain[1])).toBeCloseTo(0)
    expect(log.ticks.every(tick => Number.isInteger(Math.round(Math.log10(tick) * 1e6) / 1e6))).toBe(true)
  })
  it('keeps narrow log ranges readable and time ticks inside the range', () => {
    expect(logScale(0.06, 40, 400, 0).ticks.length).toBeGreaterThan(3)
    const time = tickedLogScale(0.05, 1045, 400, 0, [1 / 60, 1 / 6, 1, 10, 60, 600, 6000])
    expect(time.domain).toEqual([1 / 60, 6000])
    expect(yearTicks(Date.UTC(2023, 2, 1), Date.UTC(2028, 9, 1), 320).map(tick => tick.label)).toContain('2028')
  })
})
