"use strict";
// Renders results.json into whichever page is open. Charts use uPlot; everything else is plain DOM.

const METHODS = {
  distance: { label: "Distance", color: "--s1" },
  cointegration: { label: "Cointegration", color: "--s2" },
  kalman: { label: "Kalman filter", color: "--s3" },
};
const KEYS = Object.keys(METHODS);
const $ = (id) => document.getElementById(id);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const pct = (x, d = 1) => (x == null ? "–" : `${(x * 100).toFixed(d)}%`);
const num = (x, d = 2) => (x == null ? "–" : x.toFixed(d));
const signed = (x, d = 1) => (x == null ? "–" : `${x > 0 ? "+" : ""}${(x * 100).toFixed(d)}%`);
const ts = (d) => Date.parse(d) / 1000;
const longDate = (d) => new Date(d).toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" });
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

let R;
const charts = []; // [{el, build}] so they can be rebuilt on theme change

function chart(el, build) {
  if (!el) return;
  const c = { el, build };
  charts.push(c);
  draw(c);
}

function draw(c) {
  if (c.u) c.u.destroy();
  c.u = c.build(c.el.clientWidth || 600);
  if (!c.ro) {
    c.ro = new ResizeObserver(() => c.u && c.u.setSize({ width: c.el.clientWidth, height: c.u.height }));
    c.ro.observe(c.el);
  }
}

function axes(yFmt, xTime = true) {
  const a = { stroke: css("--muted"), grid: { stroke: css("--grid"), width: 1 }, ticks: { stroke: css("--grid"), width: 1 }, font: `12px ${css("--sans")}` };
  return [{ ...a, ...(xTime ? {} : { values: (u, v) => v.map((x) => `${x}`) }) }, { ...a, size: 56, values: (u, v) => v.map(yFmt) }];
}

function lines(el, xs, series, { yFmt, height = 300, time = true, extra = [], points = false }) {
  return new uPlot({
    width: el.clientWidth || 600, height,
    scales: { x: { time } },
    series: [
      time ? { value: (u, v) => (v == null ? "–" : new Date(v * 1000).toLocaleDateString("en-GB")) } : { label: "Cost (bps)" },
      ...series.map((s) => ({ label: s.label, stroke: css(s.color), width: s.width || 2, dash: s.dash, points: { show: points, size: 8 }, value: (u, v) => (v == null ? "–" : yFmt(v)) })),
      ...extra,
    ],
    axes: axes(yFmt, time),
    legend: { live: true },
    cursor: { y: false, points: { size: 8 } },
  }, [xs, ...series.map((s) => s.data), ...extra.map((e) => e.data)], el);
}

function refLine(value, n, color, label, dash = [4, 4]) {
  return { label, stroke: color, width: 1, dash, data: new Array(n).fill(value), points: { show: false }, value: () => "" };
}

/* ---------- results page ---------- */

function hero() {
  const p = R.positions[0];
  if (!p) {
    $("hero-title").textContent = "No position is open today.";
    $("hero-text").textContent = "None of the three methods sees a spread wide enough to trade at the latest close.";
    return;
  }
  const m = METHODS[p.method].label.replace(/^[A-Z](?=[a-z]+$)/, (c) => c.toLowerCase());
  const longName = p.side === 1 ? p.a_name : p.b_name;
  const shortName = p.side === 1 ? p.b_name : p.a_name;
  $("hero-title").textContent = `${p.a_name} and ${p.b_name} are ${Math.abs(p.z).toFixed(1)} standard deviations apart.`;
  $("hero-text").textContent = `That is the widest open spread at the ${longDate(R.data_end)} close. The ${m} model bought ${longName} and sold ${shortName} on ${longDate(p.opened)}, betting the gap closes. So far the trade has made ${signed(p.pnl)} on the capital set aside for the pair.` +
    (Math.abs(p.z) > 4 ? ` A spread this wide usually means the relationship has broken, not that a bargain has appeared: this is the main way pair trading loses money. The rules have no stop-loss, so the position stays open until the spread converges or its six-month window ends.` : "");
  $("hero-cap").textContent = `Spread z-score of ${p.a_name} against ${p.b_name} since the pair's cohort started trading on ${longDate(p.cohort)}. A trade opens beyond the gold lines (±${R.params.entry}) and closes when the spread crosses zero.`;
  const n = p.z_dates.length;
  chart($("hero-chart"), () => lines($("hero-chart"), p.z_dates.map(ts), [{ label: `${p.a} vs ${p.b}`, data: p.z_hist, color: "--ink", width: 2.5 }], {
    yFmt: (v) => v.toFixed(1), height: 280,
    extra: [refLine(R.params.entry, n, css("--gold"), `+${R.params.entry}`, []), refLine(-R.params.entry, n, css("--gold"), `−${R.params.entry}`, []), refLine(0, n, css("--muted"), "0")],
  }));
}

function summary() {
  const best = KEYS.reduce((a, b) => (R.methods[b].net.sharpe > R.methods[a].net.sharpe ? b : a));
  const bm = R.methods[best].net;
  const mk = R.benchmark.summary;
  const years = (Date.parse(R.data_end) - Date.parse(R.start)) / 3.156e10;
  $("intro").textContent = `${R.universe.length} large caps listed in euros, ${R.candidate_pairs} same-sector pairs, ${R.cohorts} monthly cohorts and ${years.toFixed(0)} years of out-of-sample trading since ${longDate(R.start)}. Every figure is after ${R.params.cost_bps} bps per leg and side and a ${pct(R.params.borrow, 1)} borrow fee, unless marked otherwise.`;
  const passes = KEYS.filter((k) => R.methods[k].net.dsr >= 0.95);
  $("verdict-text").textContent = `Best performer: ${METHODS[best].label}, at ${pct(bm.cagr, 1)} a year with a Sharpe ratio of ${num(bm.sharpe)}, against ${pct(mk.cagr, 1)} for simply holding the stocks. ` +
    (passes.length
      ? `${passes.map((k) => METHODS[k].label).join(" and ")} pass${passes.length === 1 ? "es" : ""} the deflated Sharpe test at 95%: the edge survives correcting for the ${R.params.trials} variants tried.`
      : `None of the three passes the deflated Sharpe test at 95% (best: ${pct(Math.max(...KEYS.map((k) => R.methods[k].net.dsr)), 0)}), so after correcting for the ${R.params.trials} variants tried the profits cannot be told apart from luck. Pair trading here is a low-return, market-neutral diversifier at best, not a money machine.`);

  const rows = [
    ["Return a year", (s) => s.net.cagr, pct, mk.cagr, "max"],
    ["Volatility", (s) => s.net.vol, pct, mk.vol, null],
    ["Sharpe ratio", (s) => s.net.sharpe, num, mk.sharpe, "max"],
    ["Sharpe before costs", (s) => s.gross.sharpe, num, null, "max"],
    ["Sortino ratio", (s) => s.net.sortino, num, mk.sortino, "max"],
    ["Worst drawdown", (s) => s.net.max_dd, (x) => pct(x, 1), mk.max_dd, "max"],
    ["t-statistic (Newey–West)", (s) => s.net.t_stat, num, mk.t_stat, "max"],
    ["Deflated Sharpe: chance the edge is real", (s) => s.net.dsr, (x) => pct(x, 0), null, "max"],
    ["Beta to the market", (s) => s.net.beta, num, 1, null],
    ["Correlation with the market", (s) => s.net.corr, num, 1, null],
    ["Months with a gain", (s) => s.net.months_up, (x) => pct(x, 0), mk.months_up, "max"],
    ["Worst month", (s) => s.net.worst_month, (x) => pct(x, 1), mk.worst_month, "max"],
  ];
  const head = `<thead><tr><th scope="col">After costs, ${R.start.slice(0, 4)}–${R.data_end.slice(0, 4)}</th>${KEYS.map((k) => `<th scope="col"><span class="key" style="background:var(${METHODS[k].color})"></span>${METHODS[k].label}</th>`).join("")}<th scope="col">Market<br><span class="sub">equal-weight, held</span></th></tr></thead>`;
  const body = rows.map(([label, get, f, mkt, rule]) => {
    const vals = KEYS.map((k) => get(R.methods[k]));
    const top = rule ? Math.max(...vals) : null;
    return `<tr><th scope="row">${label}</th>${vals.map((v) => `<td${v === top ? ' class="best"' : ""}>${f(v)}</td>`).join("")}<td>${mkt == null ? "–" : f(mkt)}</td></tr>`;
  }).join("");
  $("summary").innerHTML = head + `<tbody>${body}</tbody><caption>Bold marks the best of the three methods. The market is the equal-weighted universe with dividends, a reference point rather than a competitor: pair trading is built to have no market exposure.</caption>`;
}

let costMode = "net";
function equity() {
  const xs = R.dates.map(ts);
  const key = costMode === "net" ? "equity" : "equity_gross";
  chart($("equity-chart"), () => lines($("equity-chart"), xs, KEYS.map((k) => ({ label: METHODS[k].label, data: R.methods[k][key], color: METHODS[k].color })), { yFmt: (v) => `€${v.toFixed(2)}` }));
}

function risk() {
  const xs = R.dates.map(ts);
  chart($("dd-chart"), () => lines($("dd-chart"), xs, KEYS.map((k) => ({ label: METHODS[k].label, data: R.methods[k].drawdown, color: METHODS[k].color })), { yFmt: (v) => pct(v, 0), height: 240 }));
  const rx = R.rolling_dates.map(ts);
  chart($("rs-chart"), () => lines($("rs-chart"), rx, KEYS.map((k) => ({ label: METHODS[k].label, data: R.methods[k].rolling_sharpe, color: METHODS[k].color })), {
    yFmt: (v) => v.toFixed(1), height: 240, extra: [refLine(0, rx.length, css("--muted"), "0")],
  }));
}

function heat(v) {
  if (v == null) return "";
  const a = Math.min(Math.abs(v) / 0.1, 1) * 0.55;
  return `background:rgba(var(${v >= 0 ? "--pos" : "--neg"}),${a.toFixed(2)})`;
}

function yearly() {
  const years = Object.keys(R.methods.distance.yearly);
  const head = `<thead><tr><th scope="col">Year</th>${KEYS.map((k) => `<th scope="col">${METHODS[k].label}</th>`).join("")}<th scope="col">Market</th></tr></thead>`;
  const body = years.map((y) => {
    const cells = [...KEYS.map((k) => R.methods[k].yearly[y]), R.benchmark.yearly[y]];
    const label = y === R.data_end.slice(0, 4) ? `${y} <span class="sub">to date</span>` : y;
    return `<tr><th scope="row">${label}</th>${cells.map((v) => `<td class="heat" style="${heat(v)}">${signed(v)}</td>`).join("")}</tr>`;
  }).join("");
  $("yearly").innerHTML = head + `<tbody>${body}</tbody>`;
}

function costs() {
  const c = R.params.costs;
  const be = KEYS.map((k) => {
    const pts = R.methods[k].costs;
    for (let i = 1; i < pts.length; i++) {
      if (pts[i].sharpe <= 0 && pts[i - 1].sharpe > 0) return [k, pts[i - 1].bps + ((pts[i].bps - pts[i - 1].bps) * pts[i - 1].sharpe) / (pts[i - 1].sharpe - pts[i].sharpe)];
    }
    return [k, pts.at(-1).sharpe > 0 ? Infinity : 0];
  });
  $("costs-text").textContent = `Pair trading turns its book over many times a year, so cost is the whole game. ` +
    be.map(([k, b]) => `${METHODS[k].label} stops making money at ${b === Infinity ? `more than ${c.at(-1)}` : `about ${b.toFixed(0)}`} bps`).join("; ") +
    `. A large institution pays perhaps 3–8 bps on these stocks; a retail broker often 15–30.`;
  chart($("cost-chart"), () => lines($("cost-chart"), c, KEYS.map((k) => ({ label: METHODS[k].label, data: R.methods[k].costs.map((p) => p.sharpe), color: METHODS[k].color })), {
    yFmt: (v) => v.toFixed(1), height: 260, time: false, points: true, extra: [refLine(0, c.length, css("--muted"), "0")],
  }));
  const head = `<thead><tr><th scope="col">Sharpe ratio<br><span class="sub">entry at ± z, cost in bps</span></th>${c.map((b) => `<th scope="col">${b}</th>`).join("")}</tr></thead>`;
  const body = KEYS.map((k) => R.methods[k].entries.map((e, i) => `<tr><th scope="row">${i === 0 ? METHODS[k].label : ""} <span class="sub">${e.entry.toFixed(1)}</span></th>${e.sharpe.map((s) => `<td class="heat" style="${heat(s / 10)}">${num(s)}</td>`).join("")}</tr>`).join("")).join("");
  $("robust").innerHTML = head + `<tbody>${body}</tbody><caption>Robustness: the headline uses entry at ±${R.params.entry} and ${R.params.cost_bps} bps. The other entry levels were run once, after the protocol was fixed, and are counted in the deflated Sharpe ratio.</caption>`;
}

function trades() {
  const rows = [
    ["Trades a year", (t) => num(t.per_year, 0)],
    ["Trades that made money", (t) => pct(t.win_rate, 0)],
    ["Average trade, before costs", (t) => signed(t.mean_pnl, 2)],
    ["Median holding period", (t) => `${num(t.median_days, 0)} days`],
    ["Closed because the spread converged", (t) => pct(t.converged, 0)],
    ["Pairs found per cohort (target 20)", (t) => num(t.pairs_per_cohort, 1)],
  ];
  const head = `<thead><tr><th scope="col"></th>${KEYS.map((k) => `<th scope="col">${METHODS[k].label}</th>`).join("")}</tr></thead>`;
  const body = rows.map(([label, f]) => `<tr><th scope="row">${label}</th>${KEYS.map((k) => `<td>${f(R.methods[k].trades)}</td>`).join("")}</tr>`).join("");
  const turnover = `<tr><th scope="row">Turnover a year, × capital</th>${KEYS.map((k) => `<td>${num(R.methods[k].turnover, 1)}</td>`).join("")}</tr>`;
  $("trade-table").innerHTML = head + `<tbody>${body}${turnover}</tbody><caption>Trades that never converge are closed when their six-month trading window ends. Trade returns are on the capital assigned to one pair.</caption>`;
}

let bookFilter = "all", bookAll = false;
function book() {
  const f = $("book-filter");
  f.innerHTML = [["all", "All"], ...KEYS.map((k) => [k, METHODS[k].label])].map(([k, l]) => `<button type="button" data-k="${k}" aria-pressed="${k === bookFilter}">${l}</button>`).join("");
  f.onclick = (e) => { const k = e.target.dataset.k; if (k) { bookFilter = k; book(); } };
  const all = R.positions.filter((p) => bookFilter === "all" || p.method === bookFilter);
  const rows = bookAll ? all : all.slice(0, 15);
  $("book-more").hidden = all.length <= 15;
  $("book-more").textContent = bookAll ? "Show the 15 widest" : `Show all ${all.length}`;
  $("book-more").onclick = () => { bookAll = !bookAll; book(); };
  const head = `<thead><tr><th scope="col">Position</th><th scope="col" class="t-left">Sector</th><th scope="col" class="t-left">Method</th><th scope="col">z now</th><th scope="col">Opened</th><th scope="col">Cohorts</th><th scope="col">P&amp;L so far</th></tr></thead>`;
  const body = rows.map((p) => {
    const [l, s] = p.side === 1 ? [p.a_name, p.b_name] : [p.b_name, p.a_name];
    return `<tr><td>Long ${esc(l)}, short ${esc(s)}</td><td class="t-left">${esc(p.sector)}</td><td class="t-left"><span class="key" style="background:var(${METHODS[p.method].color})"></span>${METHODS[p.method].label}</td><td>${num(p.z, 2)}</td><td>${p.opened}</td><td>${p.cohorts}</td><td>${signed(p.pnl)}</td></tr>`;
  }).join("");
  $("book-table").innerHTML = head + `<tbody>${body || `<tr><td colspan="7">No open position for this method today.</td></tr>`}</tbody><caption>${all.length} open pairs, sorted by how wide the spread is. "Cohorts" counts the monthly cohorts holding the same pair; the row shows the newest.</caption>`;
}

/* ---------- method page ---------- */

function methodPage() {
  document.querySelectorAll("[data-k]").forEach((el) => {
    const v = el.dataset.k.split(".").reduce((o, k) => o?.[k], R);
    if (v != null) el.textContent = el.dataset.fmt === "pct" ? pct(v, 1) : v;
  });
  const bySector = {};
  R.universe.forEach((s) => (bySector[s.sector] ||= []).push(s));
  $("universe").innerHTML = `<thead><tr><th scope="col">Sector</th><th scope="col" class="t-left">Stocks</th></tr></thead><tbody>` +
    Object.entries(bySector).map(([sec, ss]) => `<tr><th scope="row">${esc(sec)} <span class="sub">${ss.length}</span></th><td class="t-left" style="white-space:normal">${ss.map((s) => `${esc(s.name)} <span class="sub">${s.ticker}</span>`).join(", ")}</td></tr>`).join("") + "</tbody>";
  $("corrections").innerHTML = `<thead><tr><th scope="col">Stock</th><th scope="col">Date</th><th scope="col" class="t-left">What was wrong</th></tr></thead><tbody>` +
    R.corrections.map((c) => `<tr><td>${esc(c.name)} <span class="sub">${c.ticker}</span></td><td>${c.date}${c.end ? ` to ${c.end}` : ""}</td><td class="t-left" style="white-space:normal">${esc(c.reason)}</td></tr>`).join("") + "</tbody>";
}

/* ---------- boot ---------- */

fetch("data/results.json")
  .then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); })
  .then((data) => {
    R = data;
    $("stamp") && ($("stamp").textContent = `Prices to ${longDate(R.data_end)} close. Rebuilt ${R.generated.replace("T", " ").replace("+00:00", " UTC")} by a scheduled job after each European trading day.`);
    if ($("summary")) {
      hero(); summary(); equity(); risk(); yearly(); costs(); trades(); book();
      document.querySelectorAll("[data-cost]").forEach((b) => b.addEventListener("click", () => {
        costMode = b.dataset.cost;
        document.querySelectorAll("[data-cost]").forEach((x) => x.setAttribute("aria-pressed", x === b));
        const c = charts.find((c) => c.el.id === "equity-chart");
        c.build = () => lines(c.el, R.dates.map(ts), KEYS.map((k) => ({ label: METHODS[k].label, data: R.methods[k][costMode === "net" ? "equity" : "equity_gross"], color: METHODS[k].color })), { yFmt: (v) => `€${v.toFixed(2)}` });
        draw(c);
      }));
    }
    if ($("universe")) methodPage();
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => charts.forEach(draw));
  })
  .catch((e) => {
    const t = $("hero-title") || $("universe");
    if (t) t.textContent = `The results file could not be loaded (${e.message}). Reload the page; if it persists, the daily build failed and the previous version will return shortly.`;
  });
