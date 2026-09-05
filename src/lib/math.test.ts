import { describe, expect, it } from 'vitest'
import { chartDateWindow, cost, duration, frontier, linear, logScale, pricePoints, project, projectCost, time } from './math'
import type { Model } from '../types'

const model: Model = {
  id: 'test/model', name: 'Fixture', provider: 'Fixture', input: 0.000001,
  output: 0.000004, context: 300000, max_output: 10000, tiers: [],
  scores: { intelligence: 50, coding: null, agentic: 0 }, weights_link: null, url: 'https://example.com',
}

describe('list-price arithmetic', () => {
  it('separates input, output, and request count', () => {
    expect(cost(model, 1000, 1000)).toBeCloseTo(5)
    expect(cost(model, 1000, 1000, 1)).toBeCloseTo(0.005)
  })
  it('uses the applicable context tier', () => {
    const tiered = { ...model, tiers: [
      { minimum: 100000, input: 0.000002, output: null },
      { minimum: 200000, input: 0.000003, output: 0.000006 },
    ] }
    expect(cost(tiered, 100000, 1000, 1)).toBeCloseTo(0.204)
    expect(cost(tiered, 200000, 1000, 1)).toBeCloseTo(0.606)
  })
  it('excludes requests that do not fit the model', () => {
    expect(cost(model, 300000, 1)).toBeNull()
    expect(cost(model, 1, 10001)).toBeNull()
    expect(cost(model, -1, 100)).toBeNull()
  })
  it('applies the highest eligible tier against base rates, not an earlier tier', () => {
    const tiered = { ...model, tiers: [
      { minimum: 10000, input: 0.000002, output: 0.000008 },
      { minimum: 20000, input: 0.000003, output: null },
    ] }
    expect(cost(tiered, 20000, 1000, 1)).toBeCloseTo(0.064)
  })
  it('keeps zero scores and excludes missing scores', () => {
    expect(pricePoints([model], 'coding', 1000, 1000)).toHaveLength(0)
    expect(pricePoints([model], 'agentic', 1000, 1000)[0].score).toBe(0)
  })
  it('finds non-dominated models and handles ties', () => {
    const points = [
      { ...model, id: 'a', cost: 1, score: 30 },
      { ...model, id: 'b', cost: 1, score: 50 },
      { ...model, id: 'c', cost: 2, score: 45 },
      { ...model, id: 'd', cost: 3, score: 60 },
    ]
    expect(frontier(points).map(m => m.id)).toEqual(['b', 'd'])
  })
})

describe('conditional futures', () => {
  it('keeps a flat path when progress stops', () => {
    expect(project(60, 730, 180, 0)).toBe(60)
  })
  it('has the specified doubling time', () => {
    expect(project(60, 360, 180, 1)).toBe(240)
    expect(project(60, 360, 180, 0.5)).toBe(120)
  })
  it('rejects invalid inputs', () => {
    expect(() => project(1, -1, 180, 1)).toThrow()
    expect(() => project(1, 365, 0, 1)).toThrow()
  })
  it('keeps price assumptions separate from measured price history', () => {
    expect(projectCost(100, 0.5, 2)).toBe(25)
    expect(projectCost(100, 0, 2)).toBe(100)
    expect(projectCost(100, 0.25, 1)).toBe(75)
    expect(() => projectCost(100, 2, 1)).toThrow()
  })
})

describe('chart units', () => {
  it('formats human task durations', () => {
    expect(duration(0.5)).toBe('30 sec')
    expect(duration(120)).toBe('2.0 hours')
    expect(duration(1440)).toBe('1.0 days')
  })
  it('maps scales with an explicit domain', () => {
    expect(linear([0, 10], [0, 100])(5)).toBe(50)
    expect(logScale([1, 100], [0, 100])(10)).toBe(50)
    expect(time('2026-01-01')).toBe(Date.UTC(2026, 0, 1))
  })
  it('keeps newly released models inside the chart after a data refresh', () => {
    const window = chartDateWindow(['2023-03-14', '2027-09-05'])
    const x = linear(window.domain, [60, 1000])
    expect(x(time('2027-09-05'))).toBeLessThan(1000)
    expect(window.years).toContain(2027)
    expect(() => chartDateWindow([])).toThrow()
  })
})
