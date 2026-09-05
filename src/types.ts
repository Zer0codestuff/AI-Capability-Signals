export type Reliability = 'p50' | 'p80'
export type Score = 'intelligence' | 'coding' | 'agentic'
export type Interval = { estimate: number; ci_low: number; ci_high: number }
export type Horizon = {
  id: string
  name: string
  release_date: string
  p50: Interval
  p80: Interval
  scaffolds: (string | null)[]
}
export type Model = {
  id: string
  name: string
  provider: string
  input: number
  output: number
  tiers: { minimum: number; input: number | null; output: number | null }[]
  context: number
  max_output: number | null
  scores: Record<Score, number | null>
  weights_link: string | null
  url: string
}
export type Trend = {
  doubling_days: number
  robust_doubling_days: number
  n: number
  start: string
  end: string
  anchor_id: string
  anchor_date: string
  anchor_minutes: number
  model_ids: string[]
  backtest: { n: number; mae_log2: number | null; baseline_mae_log2: number | null; skill_ratio: number | null }
}
export type Source = {
  id: string
  name: string
  url: string
  page: string
  description: string
  sha256: string
  retrieved_at: string
}
export type Size = {
  name: string
  epoch_name: string
  release_date: string
  total_billions: number | null
  active_billions: number | null
  notes: string
  confidence: string
  url: string
  status: string
}
export type StoryData = {
  schema_version: number
  as_of: string
  sources: Source[]
  horizons: Horizon[]
  benchmark: { version: string; reliable_range_minutes: number; excluded_versions: Record<string, number> }
  prices: Model[]
  price_coverage: {
    catalogue_rows: number
    included: number
    excluded: Record<string, number>
    benchmark_version: null
    benchmark_note: string
  }
  sizes: Size[]
  trends: Record<Reliability, Trend>
}
