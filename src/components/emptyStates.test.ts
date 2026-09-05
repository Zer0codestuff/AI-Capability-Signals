import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import snapshot from '../../public/data/story.json'
import type { StoryData } from '../types'
import Price from './Price'
import CostFuture from './CostFuture'
import Future from './Future'

describe('safe presentation when source coverage changes', () => {
  beforeEach(() => {
    vi.stubGlobal('window', { matchMedia: () => ({ matches: false }) })
  })
  afterEach(() => vi.unstubAllGlobals())

  it('keeps chart controls available when the chosen score has no models', () => {
    const data = structuredClone(snapshot) as StoryData
    data.prices.forEach(model => { model.scores.intelligence = null })
    const html = renderToStaticMarkup(createElement(Price, { data }))
    expect(html).toContain('No models fit this selection')
    expect(html).toContain('Workload')
    expect(html).toContain('Coding')
  })

  it('does not crash when no model fits the price scenario workload', () => {
    const data = structuredClone(snapshot) as StoryData
    data.prices.forEach(model => { model.context = 10 })
    const html = renderToStaticMarkup(createElement(CostFuture, { data }))
    expect(html).toContain('No priced model fits this workload')
  })

  it('reports an unavailable backtest instead of formatting null as a number', () => {
    const data = structuredClone(snapshot) as StoryData
    data.trends.p50.backtest = { n: 0, mae_log2: null, baseline_mae_log2: null, skill_ratio: null }
    const html = renderToStaticMarkup(createElement(Future, { data }))
    expect(html).toContain('too few distinct earlier release dates')
  })
})
