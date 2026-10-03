export type Unit =
  | 'eci' | 'minutes' | 'params' | 'flop' | 'usd' | 'usd_mtok' | 'ops_per_usd' | 'h100e' | 'months'

export interface Point {
  d: string
  v: number
  n: string
  o?: string
  g?: string
  lo?: number
  hi?: number
  p80?: number
  e?: number
  k?: string
  m?: string
  /** Why a value is drawn but not used: unvetted, speculative, unrated, unit, held. */
  q?: string
  /** Epoch AI confidence rating. */
  c?: string
  /** Day a price was recorded. */
  s?: string
}

export interface Interval { v: number; lo?: number; hi?: number }
export interface LineStats { n: number; rate: Interval; r2: number; doubling?: Interval }
export type Verdict = 'speeding_up' | 'slowing_down' | 'steady' | 'unknown'

export interface Backtest {
  cutoffs: number
  tests: number
  typical_error: number
  bias: number
  naive_error: number
  coverage: number
  log: boolean
}

export interface Milestone { id: string; label: string; v: number; d: string; lo: string; hi: string }
export interface BandStep { d: string; v: number; lo: number; hi: number }

export interface Trend extends LineStats {
  log: boolean
  window: { from: string; to: string; n: number }
  basis: 'full' | 'recent'
  whole: LineStats
  line: { d: string; v: number }[]
  shape: { verdict: Verdict; split?: string; early?: LineStats; late?: LineStats }
  backtest: Backtest | null
  projectable: boolean
  band: BandStep[]
  milestones: Milestone[]
}

export interface Series { id: string; label: string; kind: 'step' | 'line' | 'points'; points: Point[]; trend?: Trend | null }

export interface ChartData {
  id: string
  unit: Unit
  scale: 'log' | 'linear'
  points: Point[]
  groups: { id: string; label: string }[]
  series: Series[]
  refs: { v: number; label: string }[]
}

export interface Named { n: string; v: number; d: string; p80?: number | null; c?: string }
export interface ScaleFacts { models: number; vetted: number; set_aside: number; largest: Named; largest_confident: Named }
export interface Flag { kind: string; chart: string; name: string; detail: string }
export interface LaunchPrice { model: string; input: number; output: number; reason: string; url: string }
export interface LagFacts {
  now: number
  matched: string
  recent_average: number | null
  earlier_average: number | null
  latest_record: Point | null
}

export interface PriceLevel {
  id: string; label: string; reference: string; eci: number
  first: Named; last: Named; fold: number; records: number
}

export interface BoardRow {
  org: string; country: string; n: string; v: number; d: string; access: string
  days_on_top: number; records: number
}

export interface ExplorerRow {
  n: string; o: string; c: string; a: string; d: string; e: number
  p: number | null; pk: string | null; params: number | null; compute: number | null
}

export interface Source {
  id: string; name: string; publisher: string; page: string; license: string; description: string
  retrieved_at: string; files: { url: string; sha256: string; bytes: number }[]
}

export interface Correction {
  source: string; model: string; field: string; from: number | string; to: number | string; reason: string; applied: boolean
}

export interface Story {
  version: number
  generated_on: string
  projection_until: string
  data_through: string
  chapters: {
    intelligence: { charts: { eci: ChartData }; facts: { models: number; first: Named; last: Named; records: number } }
    tasks: {
      charts: { horizon: ChartData }
      facts: { measured: number; first: Named; last: Named; published_doubling_days: number | null; reliable_limit_minutes: number }
    }
    size: {
      charts: { params: ChartData }
      bars: { disclosure: { label: string; n: number; params: number; compute: number }[] }
      facts: ScaleFacts
    }
    compute: { charts: { compute: ChartData }; facts: ScaleFacts }
    cost: { charts: { cost: ChartData }; facts: ScaleFacts & { estimates_last_two_years: number } }
    price: {
      charts: { price: ChartData }
      facts: {
        priced: number; indexed: number; levels: PriceLevel[]; kinds: Record<string, number>
        best_first: Named | null; best_last: Named | null
      }
    }
    openness: {
      charts: { access: ChartData; access_lag: ChartData }
      facts: { lag: LagFacts; closed: Named; open: Named; counts: Record<string, number> }
    }
    race: {
      charts: { country: ChartData; country_lag: ChartData }
      board: BoardRow[]
      facts: { lag: LagFacts; us: Named; china: Named; days_tracked: number; counts: Record<string, number> }
    }
    hardware: {
      charts: { chips: ChartData; clusters: ChartData }
      facts: { chips: number; clusters: number; largest: Named; clusters_through: string }
    }
  }
  explorer: ExplorerRow[]
  sources: Source[]
  quality: { corrections: Correction[]; launch_prices: LaunchPrice[]; flags: Flag[]; database_models: number; indexed_models: number; priced_models: number }
}
