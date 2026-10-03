import { useEffect, useRef, useState, type ReactNode } from 'react'
import { Plus } from '@phosphor-icons/react/dist/ssr'

export { ArrowDown, ArrowUpRight, DownloadSimple, MagnifyingGlass } from '@phosphor-icons/react/dist/ssr'

export function Mark() {
  return <svg width="20" height="20" viewBox="0 0 25 25" fill="none" aria-hidden="true">
    <path d="M3 21V16M9 21V12M15 21V7M21 21V2" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
  </svg>
}

/* Fade up on first entry into the viewport. Content stays in the markup, so reduced
   motion and browsers without IntersectionObserver show everything at once. */
export function Reveal({ children, className = '', as: Tag = 'div' }: {
  children: ReactNode; className?: string; as?: 'div' | 'header' | 'section' | 'footer'
}) {
  const ref = useRef<HTMLElement>(null)
  const [shown, setShown] = useState(false)
  useEffect(() => {
    const node = ref.current
    if (!node || typeof IntersectionObserver === 'undefined') { setShown(true); return }
    const observer = new IntersectionObserver(entries => {
      if (entries.some(entry => entry.isIntersecting)) { setShown(true); observer.disconnect() }
    }, { rootMargin: '0px 0px -6% 0px', threshold: 0.02 })
    observer.observe(node)
    return () => observer.disconnect()
  }, [])
  return <Tag ref={ref as never} className={`reveal ${shown ? 'is-in' : ''} ${className}`}>{children}</Tag>
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

export function Fold({ title, children }: { title: string; children: ReactNode }) {
  return <details className="fold">
    <summary><span>{title}</span><Plus size={18} aria-hidden="true" /></summary>
    <div className="fold-body">{children}</div>
  </details>
}
