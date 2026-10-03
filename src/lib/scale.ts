/** Axis maths for the charts: linear, log and time scales with readable ticks. */

export interface Scale {
  (value: number): number
  ticks: number[]
  domain: [number, number]
}

function niceStep(range: number, target: number) {
  const rough = range / Math.max(1, target)
  const magnitude = 10 ** Math.floor(Math.log10(rough))
  const ratio = rough / magnitude
  return (ratio >= 5 ? 10 : ratio >= 2 ? 5 : ratio >= 1 ? 2 : 1) * magnitude
}

export function linearScale(min: number, max: number, from: number, to: number, target = 5): Scale {
  const step = niceStep(max - min || 1, target)
  const low = Math.floor(min / step) * step
  const high = Math.ceil(max / step) * step
  const ticks: number[] = []
  for (let value = low; value <= high + step / 2; value += step) ticks.push(+value.toPrecision(12))
  const scale = ((value: number) => from + ((value - low) / (high - low)) * (to - from)) as Scale
  scale.ticks = ticks
  scale.domain = [low, high]
  return scale
}

/** A log scale with hand picked ticks, for units such as time where decades read badly. */
export function tickedLogScale(min: number, max: number, from: number, to: number, candidates: number[]): Scale {
  const first = candidates.filter(v => v <= min * 1.001).pop() ?? candidates[0]
  const last = candidates.find(v => v >= max / 1.001) ?? candidates[candidates.length - 1]
  const low = Math.log10(first)
  const high = Math.log10(last)
  const scale = ((value: number) =>
    from + ((Math.log10(value) - low) / (high - low)) * (to - from)) as Scale
  scale.ticks = candidates.filter(v => v >= first && v <= last)
  scale.domain = [first, last]
  return scale
}

export function logScale(min: number, max: number, from: number, to: number, target = 6): Scale {
  let low = Math.floor(Math.log10(min) + 1e-9)
  let high = Math.ceil(Math.log10(max) - 1e-9)
  if (high === low) high += 1
  const decades = high - low
  const ticks: number[] = []
  if (decades <= 2) {
    // Too few decades for a readable axis: add the 2 and 5 marks and hug the data.
    const all: number[] = []
    for (let e = low; e <= high; e++) for (const m of [1, 2, 5]) all.push(m * 10 ** e)
    const inside = all.filter(v => v >= min / 1.001 && v <= max * 1.001)
    const first = all.filter(v => v < min / 1.001).pop() ?? all[0]
    const last = all.find(v => v > max * 1.001) ?? all[all.length - 1]
    ticks.push(first, ...inside, last)
    low = Math.log10(first)
    high = Math.log10(last)
  } else {
    const every = Math.ceil(decades / target)
    for (let e = low; e <= high; e += every) ticks.push(10 ** e)
    if (Math.log10(ticks[ticks.length - 1]) < high) high = Math.log10(ticks[ticks.length - 1]) + every
    if (high > Math.log10(ticks[ticks.length - 1])) ticks.push(10 ** high)
  }
  const scale = ((value: number) =>
    from + ((Math.log10(value) - low) / (high - low)) * (to - from)) as Scale
  scale.ticks = [...new Set(ticks)]
  scale.domain = [10 ** low, 10 ** high]
  return scale
}

/** Year ticks for a time axis given in milliseconds. */
export function yearTicks(min: number, max: number, width: number) {
  const first = new Date(min).getUTCFullYear() + 1
  const last = new Date(max).getUTCFullYear()
  const room = Math.max(2, Math.floor(width / 64))
  const every = Math.max(1, Math.ceil((last - first + 1) / room))
  const ticks: { at: number; label: string }[] = []
  for (let year = last; year >= first; year -= every) {
    ticks.unshift({ at: Date.UTC(year, 0, 1), label: String(year) })
  }
  return ticks
}
