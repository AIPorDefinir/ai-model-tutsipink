import Chart from "https://cdn.jsdelivr.net/npm/chart.js@4/auto/+esm"

const DAY = 86400000
const day = s => new Date(s + "T00:00:00").getTime()
const iso = t => new Date(t).toISOString().slice(0, 10)
const fmt = n => (n == null ? "–" : n.toFixed(2))

const data = await (await fetch("data.json")).json()
const tasks = data.tareas
const cut = day(data.corte) + DAY
// Corte del ajuste de horas (PR #64, 3 nov 2026): último día con issues cerradas al momento del ajuste.
const ajusteFecha = data.ajuste
const origH = t => (t.horas_originales != null ? t.horas_originales : t.horas)

// Modelo del profesor: valor = puntos de complejidad, costo = horas
const today = cut - DAY
const done = t => t.fin_real != null
const sum = f => tasks.reduce((acc, t) => acc + f(t), 0)
const bac = sum(t => t.puntos)
const pv = sum(t => (day(t.fin) <= today ? t.puntos : 0))
const ev = sum(t => (done(t) ? t.puntos : 0))
const costPlan = sum(t => (done(t) ? t.horas : 0))
const costReal = sum(t => (done(t) ? t.horas_reales : 0))
const spi = pv ? ev / pv : null
const cpi = costPlan ? costReal / costPlan : null

const doneAntes = t => done(t) && t.fin_real <= ajusteFecha
const doneDespues = t => done(t) && t.fin_real > ajusteFecha
const cpAntes = sum(t => (doneAntes(t) ? origH(t) : 0))
const crAntes = sum(t => (doneAntes(t) ? t.horas_reales : 0))
const cpDespues = sum(t => (doneDespues(t) ? t.horas : 0))
const crDespues = sum(t => (doneDespues(t) ? t.horas_reales : 0))
const cpiAntes = cpAntes ? crAntes / cpAntes : null
const cpiDespues = cpDespues ? crDespues / cpDespues : null

document.getElementById("corte").textContent = `Fecha de corte: ${data.corte} · ${data.repo}`

// Siglas del Plan de Valor Ganado: [nombre, qué significa]
const glossary = {
  BAC: ["Valor total", "Puntos de todo el proyecto."],
  PV: ["Valor planeado", "Puntos que ya deberíamos tener a la fecha."],
  EV: ["Valor ganado", "Puntos de lo que ya terminamos."],
  CP: ["Costo planeado", "Horas estimadas de lo terminado."],
  CR: ["Costo real", "Horas que de verdad le dedicamos a lo terminado."],
  SPI: ["Índice de cronograma", "EV / PV. Arriba de 1 vamos adelantados."],
  Desv: ["Desviación", "1 − SPI."],
  CPI: ["Índice de costo", "Costo real / costo planeado, como lo pide el profe (al revés del CPI normal). Abajo de 1 usamos menos horas."],
  "Err antes": ["Antes del ajuste", "Error de estimación (1 − horas reales / horas originales) de lo terminado hasta el corte del 3 nov. Queda como registro del error que motivó el ajuste."],
  "Err después": ["Después del ajuste", "Error de estimación (1 − horas reales / horas ajustadas) de lo terminado después del corte."],
}
const extraTerms = {
  "Puntos": "Esfuerzo en Fibonacci. Sumamos días trabajados + personas que editaron + guardados / 20 en Drive: hasta 2 → 1, 4 → 2, 6 → 3, 9 → 5, 13 → 8, más → 13. Sin historial usamos las horas planeadas.",
  "Fecha real": "Cuándo empezamos y terminamos según el historial de Drive. Con * es estimada por el orden de las actividades.",
}

const kpis = [
  ["BAC", bac + " pts"],
  ["PV", pv + " pts"],
  ["EV", ev + " pts"],
  ["CP", costPlan.toFixed(1) + " h"],
  ["CR", costReal.toFixed(1) + " h"],
  ["SPI", fmt(spi), spi >= 1],
  ["Desv", spi == null ? "–" : (100 * (1 - spi)).toFixed(0) + " %", spi >= 1],
  ["CPI", fmt(cpi), cpi == null || cpi <= 1],
  ["Err antes", cpiAntes == null ? "–" : (100 * (1 - cpiAntes)).toFixed(0) + " %", cpiAntes == null || cpiAntes <= 1],
  ["Err después", cpiDespues == null ? "–" : (100 * (1 - cpiDespues)).toFixed(0) + " %", cpiDespues == null ? null : cpiDespues <= 1],
]
document.getElementById("kpis").innerHTML = kpis
  .map(([k, v, ok]) => `<div class="kpi" title="${k} · ${glossary[k][0]}: ${glossary[k][1]}"><span>${k} <i class="info">i</i></span><strong class="${ok == null ? "" : ok ? "bien" : "mal"}">${v}</strong><small>${glossary[k][0]}</small></div>`)
  .join("")
document.getElementById("glosario").innerHTML = `<summary>¿Qué significa cada sigla?</summary><dl>${[
  ...Object.entries(glossary).map(([k, [name, desc]]) => [`${k} · ${name}`, desc]),
  ...Object.entries(extraTerms),
].map(([k, d]) => `<dt>${k}</dt><dd>${d}</dd>`).join("")}</dl>`

// Series acumuladas por fecha: planeado por fecha planeada, ganado/real por fecha real
const dates = [...new Set([ajusteFecha, ...tasks.flatMap(t => [t.fin, t.fin_real]).filter(Boolean)])].sort()
const cumulative = (f, at) => dates.map(d => (at === "plan" || d <= iso(today) ? sum(t => f(t, d)) : null))
const byPlan = v => (t, d) => (t.fin <= d ? v(t) : 0)
const byReal = v => (t, d) => (t.fin_real && t.fin_real <= d ? v(t) : 0)
const vline = (id, date, color, label, yOff = 10) => ({
  id,
  afterDatasetsDraw(c) {
    const i = dates.indexOf(date)
    if (i < 0) return
    const x = c.scales.x.getPixelForValue(i)
    const { top, bottom } = c.chartArea, g = c.ctx
    g.save(); g.strokeStyle = color; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(x, top); g.lineTo(x, bottom); g.stroke()
    g.fillStyle = color; g.font = "12px system-ui, sans-serif"
    g.fillText(label, x + 4, top + yOff); g.restore()
  },
})
const hoy = {
  id: "hoy",
  afterDatasetsDraw(c) {
    const i = dates.findIndex(d => d > iso(today))
    const x = c.scales.x.getPixelForValue(i < 0 ? dates.length - 1 : Math.max(i - 0.5, 0))
    const { top, bottom } = c.chartArea, g = c.ctx
    g.save(); g.strokeStyle = "#2563eb"; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(x, top); g.lineTo(x, bottom); g.stroke()
    g.fillStyle = "#2563eb"; g.font = "12px system-ui, sans-serif"
    g.fillText("hoy", x + 4, top + 10); g.restore()
  },
}
const corteAjuste = vline("ajuste", ajusteFecha, "#7c3aed", "ajuste", 10)
const lineChart = (id, unit, series, extraPlugins = []) => new Chart(document.getElementById(id), {
  type: "line",
  data: { labels: dates, datasets: series.map(([label, data, color, dash, extra]) => ({
    label, data, borderColor: color, backgroundColor: color, borderDash: dash || [],
    pointRadius: extra?.pointRadius ?? 2, tension: 0, spanGaps: false,
    borderWidth: extra?.borderWidth ?? 2,
    skipLegend: extra?.skipLegend ?? false,
  })) },
  options: {
    interaction: { mode: "index", intersect: false },
    scales: { y: { beginAtZero: true, title: { display: true, text: unit } } },
    plugins: { legend: { labels: { filter: (item, data) => !data.datasets[item.datasetIndex]?.skipLegend } } },
  },
  plugins: [hoy, ...extraPlugins],
})
lineChart("valor", "puntos acumulados", [
  ["PV · valor planeado", cumulative(byPlan(t => t.puntos), "plan"), "#6b7280", [6, 4]],
  ["EV · valor ganado", cumulative(byReal(t => t.puntos)), "#059669"],
])

// Proyección de costo: la línea principal se corta en el ajuste.
// Hasta el corte, horas originales. A partir de ahí, sale del acumulado real
// y suma las horas ajustadas de lo que sigue; la proyección anterior queda como sombra.
const planAnterior = dates.map(d => sum(t => (t.fin <= d ? origH(t) : 0)))
const realAlCorte = sum(t => (t.fin_real && t.fin_real <= ajusteFecha ? t.horas_reales : 0))
const pendiente = t => !t.fin_real
const planHastaCorte = dates.map((d, i) => (d <= ajusteFecha ? planAnterior[i] : null))
const planDesdeReal = dates.map(d => d < ajusteFecha
  ? null
  : realAlCorte + sum(t => (pendiente(t) && t.fin <= d ? t.horas : 0)))
const sombraAnterior = dates.map((d, i) => (d >= ajusteFecha ? planAnterior[i] : null))
lineChart("costo", "horas acumuladas", [
  ["Costo planeado", planHastaCorte, "#6b7280", [6, 4]],
  ["Costo planeado", planDesdeReal, "#6b7280", [6, 4], { skipLegend: true }],
  ["Proyección anterior", sombraAnterior, "rgba(107, 114, 128, 0.38)", [6, 4], { pointRadius: 0, borderWidth: 2 }],
  ["Costo real", cumulative(byReal(t => t.horas_reales)), "#dc2626"],
], [corteAjuste])

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

const cols = ["WBS", "Actividades", "Issue", "Responsables", "Valor planeado", "Costo planeado", "Fecha planeada", "Valor ganado", "Costo real", "Inicio real", "Fecha real"]
const late = t => t.fin_real && t.fin_real > t.fin
const rows = tasks.map(t => [
  t.codigo,
  t.definido ? t.nombre : `<em>${t.nombre} (por definir)</em>`,
  t.url ? `<a href="${t.url}" target="_blank">#${t.issue}</a>` : "–",
  t.responsables.join("<br>") || "–",
  t.puntos,
  t.horas,
  t.fin,
  done(t) ? t.puntos : 0,
  t.horas_reales,
  t.inicio_real || "–",
  t.fin_real ? `<span class="${late(t) ? "mal" : "bien"}">${t.fin_real}${t.estimada ? "*" : ""}</span>` : "–",
])
const isNum = i => i === 4 || i === 5 || i === 7 || i === 8
const colHelp = {
  WBS: "Código en la estructura de desglose del trabajo",
  "Valor planeado": "Puntos de complejidad (Fibonacci) del entregable",
  "Costo planeado": "Horas estimadas del entregable",
  "Fecha planeada": "Fin planeado (plan.json)",
  "Valor ganado": "Puntos si ya tiene fecha real, si no 0",
  "Costo real": "Horas reales registradas en el Project",
  "Fecha real": "fin_real de plan.json o fecha de cierre del issue; en rojo si es posterior a la planeada",
}
const header = `<tr>${cols.map((c, i) => `<th class="${isNum(i) ? "num" : ""}"${colHelp[c] ? ` title="${colHelp[c]}"` : ""}>${c}${colHelp[c] ? ` <i class="info">i</i>` : ""}</th>`).join("")}</tr>`
const body = Object.entries(Object.groupBy(rows, (_, i) => tasks[i].fase))
  .map(([fase, rs]) => `<tr class="fase"><td colspan="${cols.length}">${fase}</td></tr>` +
    rs.map(r => `<tr>${r.map((c, i) => `<td class="${isNum(i) ? "num" : ""}">${c}</td>`).join("")}</tr>`).join(""))
  .join("")
document.getElementById("tabla").innerHTML = header + body
