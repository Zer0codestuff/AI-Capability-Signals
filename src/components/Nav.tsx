import { useEffect, useState } from 'react'
import { Mark } from './ui'

export const CHAPTERS = [
  { id: 'progress', label: 'Progress' },
  { id: 'cost', label: 'Cost' },
  { id: 'scale', label: 'Scale' },
  { id: 'future', label: 'Future' },
] as const

export default function Nav({ active }: { active: string }) {
  const [open, setOpen] = useState(false)
  useEffect(() => {
    if (!open) return
    const previous = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    const onKey = (event: KeyboardEvent) => { if (event.key === 'Escape') setOpen(false) }
    window.addEventListener('keydown', onKey)
    return () => { document.body.style.overflow = previous; window.removeEventListener('keydown', onKey) }
  }, [open])

  const links = (onPick?: () => void) => <>
    {CHAPTERS.map((chapter, i) =>
      <a key={chapter.id} href={`#${chapter.id}`} onClick={onPick}
        aria-current={active === chapter.id ? 'location' : undefined}
        style={{ transitionDelay: onPick ? `${100 + i * 50}ms` : undefined }}>
        <span className="nav-number" aria-hidden="true">{`0${i + 1}`}</span>{chapter.label}
      </a>)}
    <a href="#faq" onClick={onPick} aria-current={active === 'faq' ? 'location' : undefined}
      style={{ transitionDelay: onPick ? '300ms' : undefined }}>
      <span className="nav-number" aria-hidden="true">05</span>Questions</a>
    <a href="#sources" onClick={onPick} className="nav-sources"
      aria-current={active === 'sources' ? 'location' : undefined}
      style={{ transitionDelay: onPick ? '350ms' : undefined }}>Sources</a>
  </>

  return <>
    <header className={`island ${open ? 'is-open' : ''}`}>
      <a className="brand" href="#top" aria-label="AI Capability Signals, back to top" onClick={() => setOpen(false)}>
        <Mark /><span>AI Capability Signals</span>
      </a>
      <nav className="island-links" aria-label="Story chapters">{links()}</nav>
      <button type="button" className="burger" aria-expanded={open} aria-controls="chapter-menu"
        aria-label={open ? 'Close the chapter menu' : 'Open the chapter menu'} onClick={() => setOpen(v => !v)}>
        <span /><span />
      </button>
    </header>
    <div id="chapter-menu" className={`menu ${open ? 'is-open' : ''}`} inert={!open}>
      <nav aria-label="Story chapters, expanded">{links(() => setOpen(false))}</nav>
    </div>
  </>
}
