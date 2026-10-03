import { useEffect, useState } from 'react'
import { Mark } from './ui'

export const SECTIONS = [
  { id: 'capability', label: 'Capability' },
  { id: 'price', label: 'Price' },
  { id: 'scale', label: 'Scale' },
  { id: 'race', label: 'Race' },
  { id: 'models', label: 'Models' },
  { id: 'method', label: 'Method' },
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

  const links = (onPick?: () => void) => SECTIONS.map((section, i) =>
    <a key={section.id} href={`#${section.id}`} onClick={onPick}
      aria-current={active === section.id ? 'location' : undefined}
      style={{ transitionDelay: onPick ? `${80 + i * 40}ms` : undefined }}>{section.label}</a>)

  return <>
    <header className={`island ${open ? 'is-open' : ''}`}>
      <a className="brand" href="#top" aria-label="AI Capability Signals, back to top" onClick={() => setOpen(false)}>
        <Mark /><span>AI Capability Signals</span>
      </a>
      <nav className="island-links" aria-label="Sections">{links()}</nav>
      <button type="button" className="burger" aria-expanded={open} aria-controls="section-menu"
        aria-label={open ? 'Close the menu' : 'Open the menu'} onClick={() => setOpen(v => !v)}>
        <span /><span />
      </button>
    </header>
    <div id="section-menu" className={`menu ${open ? 'is-open' : ''}`} inert={!open}>
      <nav aria-label="Sections, expanded">{links(() => setOpen(false))}</nav>
    </div>
  </>
}
