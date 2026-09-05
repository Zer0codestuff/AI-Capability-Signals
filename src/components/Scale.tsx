import { useState } from 'react'
import type { StoryData } from '../types'
import { Evidence, SectionHead, Segment, SourceLine } from './ui'

export default function Scale({ data }: { data: StoryData }) {
  const [mode, setMode] = useState<'total' | 'active'>('total')
  const maximum = Math.max(...data.sizes.map(m => m.total_billions ?? 0))
  return <section id="scale" className="chapter">
    <SectionHead number="03" label="The scale" title={<>A bigger model.<br />A smarter model?</>}>
      Parameters are the numbers a model learns. They describe its size, not its intelligence.
      And sometimes we simply do not know how many there are.
    </SectionHead>
    <div className="chart-toolbar"><span className="chart-label">Billions of parameters</span>
      <Segment value={mode} onChange={setMode} label="Parameter measure" options={[
        { value: 'total', label: 'Total size' }, { value: 'active', label: 'Active per token' },
      ]} />
    </div>
    <div className="size-chart" aria-live="polite">
      {data.sizes.map(model => {
        const value = mode === 'total' ? model.total_billions : model.active_billions
        return <div className="size-row" key={model.name}>
          <div className="size-name"><a href={model.url} target="_blank" rel="noreferrer">{model.name}</a>
            <span>{model.release_date.slice(0, 4)}</span></div>
          <div className={`size-track ${value === null ? 'unknown-track' : ''}`}>
            {value !== null ? <div className="size-fill" style={{ width: `${value / maximum * 100}%` }} />
              : <span className="unknown-label">{mode === 'total' ? 'Not in the source' : 'No separate active count in this record'}</span>}
          </div>
          <strong className={value === null ? 'unknown-value' : ''}>{value !== null ? `${value}B` : '?'}</strong>
        </div>
      })}
    </div>
    <div className="insight-line"><span aria-hidden="true">↳</span><p>
      {mode === 'total'
        ? 'An unknown number is not a small number. Missing specifications stay missing.'
        : 'Some models only activate a fraction of their parameters for each token. Total size alone misses this.'}
    </p></div>
    <SourceLine href="https://epoch.ai/data/ai-models" label="Epoch AI + linked model reports">
      Four illustrative releases, not a ranking of the latest models · fixed linear scale
    </SourceLine>
    <Evidence title="Why there is no parameter forecast">
      <p>These are selected examples of dense models, mixture-of-experts models, and unavailable
        specifications. Qwen3’s 22B and DeepSeek V3’s 37B active counts come from the published
        parameter notes. A missing active count is not filled with the total count.</p>
      <p>Epoch records can contain reported values, derived estimates, or speculation. A populated
        field is not proof that a developer disclosed it. Each example links to its supporting
        publication; the downloaded dataset preserves the notes and Epoch confidence labels.</p>
      <p>Architecture, training data, post-training and inference effort also affect performance.
        Extrapolating a parameter curve would not tell us how capable GPT-7, GPT-8 or GPT-9 will be.</p>
    </Evidence>
  </section>
}
