import Chart from "https://cdn.jsdelivr.net/npm/chart.js@4/auto/+esm"

const DAY = 86400000
const day = s => new Date(s + "T00:00:00").getTime()
const iso = t => new Date(t).toISOString().slice(0, 10)
const fmt = n => (n == null ? "–" : n.toFixed(2))

const data = await (await fetch("data.json")).json()
const tasks = data.tareas
const cut = day(data.corte) + DAY

const planned = (t, at) => {
  const start = day(t.inicio)
  const end = day(t.fin) + DAY
  if (at <= start) return 0
  if (at >= end) return t.horas
  return t.horas * (at - start) / (end - start)
}
const sum = f => tasks.reduce((acc, t) => acc + f(t), 0)

const bac = sum(t => t.horas)
const pv = sum(t => planned(t, cut))
const ev = sum(t => t.horas * t.avance)
const ac = sum(t => t.horas_reales)
const spi = pv ? ev / pv : null
const cpi = ac ? ev / ac : null

document.getElementById("corte").textContent = `Fecha de corte: ${data.corte} · ${data.repo}`

const kpis = [
  ["BAC", bac.toFixed(1) + " h"],
  ["PV", pv.toFixed(1) + " h"],
  ["EV", ev.toFixed(1) + " h"],
  ["AC", ac.toFixed(1) + " h"],
  ["SV", (ev - pv).toFixed(1) + " h", ev >= pv],
  ["CV", (ev - ac).toFixed(1) + " h", ev >= ac],
  ["SPI", fmt(spi), spi >= 1],
  ["CPI", fmt(cpi), cpi == null || cpi >= 1],
]
document.getElementById("kpis").innerHTML = kpis
  .map(([k, v, ok]) => `<div class="kpi"><span>${k}</span><strong class="${ok == null ? "" : ok ? "bien" : "mal"}">${v}</strong></div>`)
  .join("")

const first = Math.min(...tasks.map(t => day(t.inicio)))
const last = Math.max(...tasks.map(t => day(t.fin)))
const days = []
for (let t = first; t <= last; t += DAY) days.push(t)
const cutIndex = days.findIndex(t => t + DAY >= cut)
const marker = v => days.map((_, i) => (i === cutIndex ? v : null))

new Chart(document.getElementById("curva"), {
  type: "line",
  data: {
    labels: days.map(iso),
    datasets: [
      { label: "PV (planificado)", data: days.map(t => sum(x => planned(x, t + DAY))), borderColor: "#6b7280", pointRadius: 0 },
      { label: "EV (ganado)", data: marker(ev), borderColor: "#059669", backgroundColor: "#059669", pointRadius: 7 },
      { label: "AC (real)", data: marker(ac), borderColor: "#dc2626", backgroundColor: "#dc2626", pointRadius: 7, pointStyle: "rectRot" },
    ],
  },
  options: { scales: { y: { beginAtZero: true, title: { display: true, text: "horas acumuladas" } } } },
})

const colors = { hecha: "#059669", encurso: "#d97706", pendiente: "#6b7280", sinissue: "#d1d5db" }
const status = t => (!t.issue ? "sinissue" : t.avance >= 1 ? "hecha" : t.avance > 0 ? "encurso" : "pendiente")
const phases = Object.groupBy(tasks, t => t.fase)
const byCode = Object.fromEntries(tasks.map(t => [t.codigo, t]))

const hours = (plan, real, pct) => `${plan} h plan · ${+real.toFixed(1)} h real · ${pct}%`

const summary = ts => {
  const h = ts.reduce((a, t) => a + t.horas, 0)
  const e = ts.reduce((a, t) => a + t.horas * t.avance, 0)
  const r = ts.reduce((a, t) => a + t.horas_reales, 0)
  return hours(h, r, Math.round(100 * e / h))
}

const leaf = t => {
  const tag = t.url ? "a" : "div"
  const href = t.url ? ` href="${t.url}" target="_blank"` : ""
  return `<li><${tag}${href} class="caja hoja ${status(t)}" data-codigo="${t.codigo}" title="${t.inicio} → ${t.fin}&#10;Responsables: ${t.responsables.join(", ") || "–"}&#10;Depende de: ${t.depende.join(", ") || "–"}">
    ${t.codigo} ${t.nombre}
    <small>${hours(t.horas, t.horas_reales, Math.round(t.avance * 100))}</small>
    <span class="barra" style="width:${t.avance * 100}%"></span>
  </${tag}></li>`
}

const activity = ([name, ts]) =>
  `<li><div class="caja">${name}<small>${summary(ts)}</small></div><ul>${ts.map(leaf).join("")}</ul></li>`

const phase = ([name, ts]) =>
  `<div class="columna"><div class="caja">${name}<small>${summary(ts)}</small></div>
  <ul>${Object.entries(Object.groupBy(ts, t => t.actividad)).map(activity).join("")}</ul></div>`

const legend = [["hecha", "Hecha"], ["encurso", "En curso"], ["pendiente", "Pendiente"], ["sinissue", "Sin issue"]]
  .map(([k, v]) => `<span><i style="background:${colors[k]}"></i>${v}</span>`).join("")

document.getElementById("grafo").innerHTML = `
  <div class="leyenda">${legend}<span>Pasa el cursor sobre un entregable: <b class="t-previa">depende de</b> · <b class="t-sigue">lo necesitan</b></span></div>
  <div class="wbs">
    <div class="caja raiz">${data.proyecto}<small>${summary(tasks)}</small></div>
    <div class="columnas">${Object.entries(phases).map(phase).join("")}</div>
  </div>`

const boxes = [...document.querySelectorAll(".hoja")]
for (const box of boxes) {
  const t = byCode[box.dataset.codigo]
  box.addEventListener("mouseenter", () => boxes.forEach(o => {
    o.classList.toggle("previa", t.depende.includes(o.dataset.codigo))
    o.classList.toggle("sigue", byCode[o.dataset.codigo].depende.includes(t.codigo))
  }))
  box.addEventListener("mouseleave", () => boxes.forEach(o => o.classList.remove("previa", "sigue")))
}

const cols = ["WBS", "Entregable", "Issue", "Responsables", "Inicio", "Fin", "Horas", "Avance", "EV", "Reales", "Depende de"]
const rows = tasks.map(t => [
  t.codigo,
  t.definido ? t.nombre : `<em>${t.nombre} (por definir)</em>`,
  t.url ? `<a href="${t.url}" target="_blank">#${t.issue}</a>` : "–",
  t.responsables.join("<br>") || "–",
  t.inicio,
  t.fin,
  t.horas,
  Math.round(t.avance * 100) + "%",
  (t.horas * t.avance).toFixed(1),
  t.horas_reales,
  t.depende.join(", "),
])
const isNum = i => i >= 6 && i <= 9
const header = `<tr>${cols.map((c, i) => `<th class="${isNum(i) ? "num" : ""}">${c}</th>`).join("")}</tr>`
const body = Object.entries(Object.groupBy(rows, (_, i) => tasks[i].fase))
  .map(([fase, rs]) => `<tr class="fase"><td colspan="${cols.length}">${fase}</td></tr>` +
    rs.map(r => `<tr>${r.map((c, i) => `<td class="${isNum(i) ? "num" : ""}">${c}</td>`).join("")}</tr>`).join(""))
  .join("")
document.getElementById("tabla").innerHTML = header + body
