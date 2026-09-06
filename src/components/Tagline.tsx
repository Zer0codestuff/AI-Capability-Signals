import { useEffect, useRef } from 'react'

/* Large statement whose words switch from muted to full ink as each crosses a trigger line
   a little above the middle of the viewport. Words that cross together are staggered in
   reading order so the line lights up left to right rather than flipping at once. */
export default function Tagline({ lines, label }: { lines: string[]; label: string }) {
  const ref = useRef<HTMLParagraphElement>(null)
  useEffect(() => {
    const root = ref.current
    if (!root || typeof IntersectionObserver === 'undefined') return
    const words = Array.from(root.querySelectorAll<HTMLElement>('.word'))
    const observer = new IntersectionObserver(entries => {
      entries.filter(entry => entry.isIntersecting)
        .sort((a, b) => Number((a.target as HTMLElement).dataset.index) - Number((b.target as HTMLElement).dataset.index))
        .forEach((entry, i) => {
          const word = entry.target as HTMLElement
          word.style.transitionDelay = `${i * 45}ms`
          word.classList.add('is-on')
          observer.unobserve(word)
        })
    }, { rootMargin: '0px 0px -42% 0px', threshold: 1 })
    words.forEach(word => observer.observe(word))
    // A hash jump can move the whole statement past the trigger line without any word
    // crossing it. Once the paragraph has scrolled out above the viewport, light everything.
    const exit = new IntersectionObserver(entries => {
      const entry = entries[0]
      if (!entry.isIntersecting && entry.boundingClientRect.top < 0) {
        words.forEach(word => word.classList.add('is-on'))
        observer.disconnect(); exit.disconnect()
      }
    })
    exit.observe(root)
    return () => { observer.disconnect(); exit.disconnect() }
  }, [])
  let index = 0
  return <section className="tagline" aria-label={label}>
    <p ref={ref} className="tagline-text">
      {lines.map((line, l) => <span className="tagline-line" key={l}>
        {line.split(' ').map((word, w) =>
          <span className="word" key={w} data-index={index++}>{word}{w < line.split(' ').length - 1 ? ' ' : ''}</span>)}
      </span>)}
    </p>
  </section>
}
