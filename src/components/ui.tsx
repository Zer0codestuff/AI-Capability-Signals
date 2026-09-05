import type { ReactNode } from 'react'

export function Arrow({ down = false }: { down?: boolean }) {
  return <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true"
    className={down ? 'arrow-down' : ''}>
    <path d="M5 19 19 5M5 5h14v14" stroke="currentColor" strokeWidth="1.5" />
  </svg>
}

export function Mark() {
  return <svg width="25" height="25" viewBox="0 0 25 25" fill="none" aria-hidden="true">
    <path d="M3 21V16M9 21V12M15 21V7M21 21V2" stroke="currentColor" strokeWidth="2.5" />
  </svg>
}

export function SectionHead({ number, label, title, children }: {
  number: string; label: string; title: ReactNode; children: ReactNode
}) {
  return <header className="section-head">
    <div className="eyebrow"><span className="chapter-number">{number}</span>{label}</div>
    <div className="section-intro"><h2>{title}</h2><p>{children}</p></div>
  </header>
}

export function Evidence({ title = 'How to read this', children }: { title?: string; children: ReactNode }) {
  return <details className="evidence">
    <summary>{title}<span aria-hidden="true">+</span></summary>
    <div className="evidence-body">{children}</div>
  </details>
}

export function Segment<T extends string>({ value, options, onChange, label }: {
  value: T; options: readonly { value: T; label: string }[]
  onChange: (value: T) => void; label: string
}) {
  return <div className="segment" role="group" aria-label={label}>
    {options.map(option => <button key={option.value} type="button"
      aria-pressed={value === option.value} onClick={() => onChange(option.value)}>
      {option.label}
    </button>)}
  </div>
}

export function SourceLine({ children, href, label }: { children: ReactNode; href: string; label: string }) {
  return <div className="source-line"><span>{children}</span>
    <a href={href} target="_blank" rel="noreferrer">{label}<Arrow /></a>
  </div>
}
