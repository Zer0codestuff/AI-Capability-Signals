import { useEffect, useRef, useState, type ReactNode } from 'react'
import { ArrowUpRight, Plus } from '@phosphor-icons/react/dist/ssr'

export { ArrowUpRight, ArrowDown, ArrowRight, ArrowUp, ArrowCounterClockwise, DownloadSimple, Diamond }
  from '@phosphor-icons/react/dist/ssr'

export function Mark() {
  return <svg width="20" height="20" viewBox="0 0 25 25" fill="none" aria-hidden="true">
    <path d="M3 21V16M9 21V12M15 21V7M21 21V2" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
  </svg>
}

/* Heavy fade up on first entry into the viewport. Content stays in the markup, only the
   presentation is delayed, so server rendering and reduced motion keep everything visible. */
export function Reveal({ children, className = '', as: Tag = 'div', delay = 0 }: {
  children: ReactNode; className?: string; as?: 'div' | 'header' | 'section' | 'footer'; delay?: number
}) {
  const ref = useRef<HTMLElement>(null)
  const [shown, setShown] = useState(false)
  useEffect(() => {
    const node = ref.current
    if (!node || typeof IntersectionObserver === 'undefined') { setShown(true); return }
    const observer = new IntersectionObserver(entries => {
      if (entries.some(entry => entry.isIntersecting)) { setShown(true); observer.disconnect() }
    }, { rootMargin: '0px 0px -8% 0px', threshold: 0.05 })
    observer.observe(node)
    return () => observer.disconnect()
  }, [])
  return <Tag ref={ref as never} className={`reveal ${shown ? 'is-in' : ''} ${className}`}
    style={delay ? { transitionDelay: `${delay}ms` } : undefined}>{children}</Tag>
}

export function SectionHead({ number, label, title, children }: {
  number: string; label: string; title: ReactNode; children: ReactNode
}) {
  return <Reveal as="header" className="section-head">
    <p className="eyebrow"><span className="chapter-number" aria-hidden="true">{number}</span>{label}</p>
    <div className="section-intro"><h2>{title}</h2><p>{children}</p></div>
  </Reveal>
}

export function Evidence({ title = 'How to read this', children }: { title?: string; children: ReactNode }) {
  return <details className="evidence">
    <summary><span>{title}</span><Plus size={18} weight="regular" aria-hidden="true" /></summary>
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
    <a href={href} target="_blank" rel="noreferrer">{label}<ArrowUpRight size={14} aria-hidden="true" /></a>
  </div>
}

export function Field({ label, children, className = '' }: { label: string; children: ReactNode; className?: string }) {
  return <label className={`field ${className}`}><span>{label}</span>{children}</label>
}
