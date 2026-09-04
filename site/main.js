/* nodewatch showcase site — hero trace animation.
 *
 * One requestAnimationFrame driver walks a declarative timeline and derives every
 * bit of visual state from a single elapsed-time value. Nothing accumulates, so
 * the animation can't drift, replay is just `t = 0`, and the reduced-motion path
 * is `t = END` rendered once.
 *
 * The SVG is decorative (aria-hidden); the <table> beside it carries the
 * semantics and is populated before any animation runs, so the content exists
 * for assistive tech rather than being injected at the end.
 */
(function () {
  "use strict";

  /* The documented example run from the project README, so this page cannot
     drift from the product's own numbers. Per-node `tokens` is input+output. */
  var TRACE = {
    nodes: [
      { name: "planner",    tokens: 9600,  loops: 1, tools: 0, cost: 0.23 },
      { name: "retriever",  tokens: 0,     loops: 1, tools: 3, cost: 0.00 },
      { name: "analyzer",   tokens: 25900, loops: 2, tools: 1, cost: 0.42 },
      { name: "summarizer", tokens: 9700,  loops: 1, tools: 0, cost: 0.20 }
    ],
    totalTokens: 45200,
    totalCost: 0.85,
    totalSeconds: 12.3
  };

  var RAMP = { azure: "#58A6FF", amber: "#F2A93B", coral: "#FF6A5A", dim: "#7C8AA6" };
  var reduced = window.matchMedia("(prefers-reduced-motion: reduce)");

  /* ── colour is data ────────────────────────────────────────────────────
     A node's colour is its share of the bill, not a theme choice: cheap runs
     azure, ~30% amber, ~50%+ coral. Zero-cost nodes stay dim. */
  function lerp(a, b, t) { return a + (b - a) * t; }
  function hex2rgb(h) {
    return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
  }
  function mix(h1, h2, t) {
    var a = hex2rgb(h1), b = hex2rgb(h2);
    return "rgb(" + Math.round(lerp(a[0], b[0], t)) + "," +
                    Math.round(lerp(a[1], b[1], t)) + "," +
                    Math.round(lerp(a[2], b[2], t)) + ")";
  }
  function costColor(share) {
    if (share <= 0) return RAMP.dim;
    if (share < 0.3) return mix(RAMP.azure, RAMP.amber, share / 0.3);
    return mix(RAMP.amber, RAMP.coral, Math.min(1, (share - 0.3) / 0.2));
  }

  var maxCost = TRACE.nodes.reduce(function (m, n) { return Math.max(m, n.cost); }, 0);
  TRACE.nodes.forEach(function (n) {
    n.share = TRACE.totalCost ? n.cost / TRACE.totalCost : 0;
    n.color = costColor(n.share);
  });

  /* ── timeline: absolute windows, so any t maps to a complete frame ────── */
  var D = { wake: 350, edge: 240, ignite: 130, count: 700, loop: 500, settle: 650 };
  var LOOP_LEN = 96, HEAD_LEN = 14;

  var phases = [];
  (function buildTimeline() {
    var cursor = D.wake;
    TRACE.nodes.forEach(function (n, i) {
      var edgeStart = cursor;
      var edgeEnd = i === 0 ? cursor : cursor + D.edge;
      var igniteAt = edgeEnd;
      var countStart = igniteAt + D.ignite;
      var countEnd = countStart + D.count;
      var loopStart = n.loops > 1 ? countEnd - 120 : 0;
      var loopEnd = n.loops > 1 ? loopStart + D.loop : 0;
      phases.push({
        node: n, edgeStart: edgeStart, edgeEnd: edgeEnd, igniteAt: igniteAt,
        countStart: countStart, countEnd: countEnd, loopStart: loopStart, loopEnd: loopEnd
      });
      cursor = Math.max(countEnd, loopEnd);
    });
    phases.runEnd = cursor;
  })();
  var END = phases.runEnd + D.settle;

  function clamp01(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
  function easeOut(t) { return 1 - Math.pow(1 - t, 3); }
  function prog(t, a, b) { return b <= a ? (t >= b ? 1 : 0) : clamp01((t - a) / (b - a)); }
  function fmtInt(n) { return Math.round(n).toLocaleString("en-US"); }
  function fmtUsd(n) { return "$" + n.toFixed(2); }

  /* ── build the SVG graph ──────────────────────────────────────────────── */
  var SVGNS = "http://www.w3.org/2000/svg";
  var BOX = { x: 22, w: 210, h: 56, gap: 40 };
  var svg = document.getElementById("graph");
  var gfx = [];

  function el(name, attrs) {
    var e = document.createElementNS(SVGNS, name);
    for (var k in attrs) {
      if (Object.prototype.hasOwnProperty.call(attrs, k)) e.setAttribute(k, attrs[k]);
    }
    return e;
  }

  function buildGraph() {
    if (!svg) return;
    var mid = BOX.x + BOX.w / 2;
    var y = 10;

    TRACE.nodes.forEach(function (n, i) {
      var parts = { pips: [] };

      if (i > 0) {
        var edge = el("path", { "class": "edge", d: "M " + mid + " " + (y - BOX.gap) + " L " + mid + " " + y });
        edge.style.setProperty("--len", String(BOX.gap));
        svg.appendChild(edge);
        parts.edge = edge;

        var head = el("path", {
          "class": "edge",
          d: "M " + (mid - 4) + " " + (y - 7) + " L " + mid + " " + y + " L " + (mid + 4) + " " + (y - 7)
        });
        head.style.setProperty("--len", String(HEAD_LEN));
        svg.appendChild(head);
        parts.head = head;
      }

      parts.halo = el("rect", { "class": "halo", x: BOX.x - 5, y: y - 5, width: BOX.w + 10, height: BOX.h + 10, rx: 12 });
      svg.appendChild(parts.halo);

      parts.box = el("rect", { "class": "nbox", x: BOX.x, y: y, width: BOX.w, height: BOX.h, rx: 8 });
      svg.appendChild(parts.box);

      /* left ignition bar — widens with the count-up, so the bar IS the number */
      parts.bar = el("rect", { "class": "nbar", x: BOX.x, y: y + 1, width: 3, height: BOX.h - 2, rx: 1.5 });
      svg.appendChild(parts.bar);

      parts.name = el("text", { "class": "nname", x: BOX.x + 14, y: y + 23 });
      parts.name.textContent = n.name;
      svg.appendChild(parts.name);

      parts.val = el("text", { "class": "nval", x: BOX.x + 14, y: y + 41 });
      parts.val.textContent = "0 tok";
      svg.appendChild(parts.val);

      /* tool-call pips: a node can cost latency without costing tokens */
      for (var p = 0; p < n.tools; p++) {
        var pip = el("rect", {
          "class": "pip", x: BOX.x + BOX.w - 16 - p * 9, y: y + BOX.h - 16, width: 5, height: 5, rx: 1
        });
        svg.appendChild(pip);
        parts.pips.push(pip);
      }

      /* loop arc, drawn only where the node actually re-entered */
      if (n.loops > 1) {
        var cx = BOX.x + BOX.w, cy = y + BOX.h / 2;
        parts.loop = el("path", {
          "class": "loop",
          d: "M " + (cx - 2) + " " + (cy - 14) +
             " C " + (cx + 26) + " " + (cy - 20) + ", " + (cx + 26) + " " + (cy + 20) + ", " +
             (cx - 2) + " " + (cy + 14)
        });
        parts.loop.style.setProperty("--len", String(LOOP_LEN));
        svg.appendChild(parts.loop);

        /* badge sits clear of the arc's bulge, not on top of it */
        parts.loopLabel = el("text", { "class": "nloop", x: cx + 26, y: cy + 4.5 });
        svg.appendChild(parts.loopLabel);
      }

      /* Transparent hover target. Deliberately NOT focusable: the whole SVG is
         aria-hidden because the table beside it is the accessible twin carrying
         the same numbers, and a focusable child inside an aria-hidden subtree is
         a real contradiction (axe: aria-hidden-focus). Keyboard and screen-reader
         users get the table, which loses nothing — the highlight is only a
         convenience for pointer users. */
      parts.hit = el("rect", {
        "class": "nhit", x: BOX.x - 5, y: y - 5, width: BOX.w + 10, height: BOX.h + 10
      });
      svg.appendChild(parts.hit);

      gfx.push(parts);
      y += BOX.h + BOX.gap;
    });

    svg.setAttribute("viewBox", "0 0 300 " + (y - BOX.gap + 10));
  }

  /* ── build the ledger table (exists before any animation) ─────────────── */
  var rows = [];

  function buildLedger() {
    var body = document.getElementById("ledger-body");
    if (!body) return;
    TRACE.nodes.forEach(function (n) {
      var tr = document.createElement("tr");

      var th = document.createElement("th");
      th.scope = "row";
      th.textContent = n.name;
      tr.appendChild(th);

      var tdTok = document.createElement("td");
      tdTok.className = "num";
      tr.appendChild(tdTok);

      var tdLoop = document.createElement("td");
      tdLoop.className = "num";
      tdLoop.textContent = String(n.loops);
      tr.appendChild(tdLoop);

      var tdCost = document.createElement("td");
      tdCost.className = "num cost-cell";
      var bar = document.createElement("span");
      bar.className = "cost-bar";
      bar.style.setProperty("--c", n.color);
      var num = document.createElement("span");
      num.className = "cost-num";
      tdCost.appendChild(bar);
      tdCost.appendChild(num);
      tr.appendChild(tdCost);

      body.appendChild(tr);
      rows.push({ tr: tr, tok: tdTok, cost: num, bar: bar, node: n });
    });
  }

  /* ── render one frame from a single elapsed time ───────────────────────── */
  var tTokens = document.getElementById("t-tokens");
  var tCost = document.getElementById("t-cost");
  var tTime = document.getElementById("t-time");
  var verdict = document.getElementById("verdict");
  var hovering = {};

  function render(t) {
    var tokSum = 0, costSum = 0;

    phases.forEach(function (ph, i) {
      var n = ph.node, g = gfx[i], r = rows[i];
      if (!g || !r) return;

      var lit = t >= ph.igniteAt;
      var c = n.cost > 0 ? n.color : RAMP.dim;
      var k = easeOut(prog(t, ph.countStart, ph.countEnd));

      if (g.edge) {
        var ek = prog(t, ph.edgeStart, ph.edgeEnd);
        g.edge.classList.toggle("lit", ek > 0);
        g.edge.style.setProperty("--c", c);
        g.edge.style.strokeDashoffset = String(BOX.gap * (1 - ek));
        g.head.classList.toggle("lit", ek >= 1);
        g.head.style.setProperty("--c", c);
        g.head.style.strokeDashoffset = String(HEAD_LEN * (1 - ek));
      }

      [g.box, g.bar, g.name, g.val].forEach(function (e) {
        e.classList.toggle("lit", lit);
        e.style.setProperty("--c", c);
      });
      g.bar.setAttribute("width", String(3 + (lit ? 4 * k : 0)));

      g.halo.style.setProperty("--c", c);
      if (!hovering[i]) g.halo.style.opacity = lit ? String(0.5 * (1 - k)) : "0";

      var tok = n.tokens * (lit ? k : 0);
      var cost = n.cost * (lit ? k : 0);
      tokSum += tok;
      costSum += cost;

      g.val.textContent = n.tokens === 0
        ? (lit ? n.tools + (n.tools === 1 ? " tool call" : " tool calls") : "0 tok")
        : fmtInt(tok) + " tok   " + fmtUsd(cost);

      g.pips.forEach(function (pip, pi) {
        pip.style.opacity = t >= ph.countStart + pi * 90 ? "1" : "0";
        pip.style.fill = c;
      });

      if (g.loop) {
        var lk = prog(t, ph.loopStart, ph.loopEnd);
        g.loop.style.opacity = lk > 0 ? "1" : "0";
        g.loop.style.strokeDashoffset = String(LOOP_LEN * (1 - lk));
        g.loopLabel.textContent = lk > 0.5 ? "↻" + n.loops : "";
      }

      r.tr.classList.toggle("lit", lit);
      r.tok.textContent = n.tokens === 0 ? "—" : fmtInt(tok);
      r.cost.textContent = fmtUsd(cost);
      r.bar.style.setProperty("--w", (maxCost ? (cost / maxCost) * 100 : 0) + "%");
    });

    tTokens.textContent = fmtInt(tokSum);
    tCost.textContent = fmtUsd(costSum);
    tTime.textContent = (TRACE.totalSeconds * clamp01(t / phases.runEnd)).toFixed(1) + "s";

    var done = t >= END - 40;
    if (verdict.hidden === done) verdict.hidden = !done;
  }

  /* ── driver ───────────────────────────────────────────────────────────── */
  var raf = null, t0 = null;

  function stop() { if (raf !== null) { cancelAnimationFrame(raf); raf = null; } }

  function play() {
    stop();
    t0 = null;
    (function step(ts) {
      if (t0 === null) t0 = ts;
      var t = ts - t0;
      render(Math.min(t, END));
      raf = t < END ? requestAnimationFrame(step) : null;
    })(performance.now());
  }

  function showFinal() { stop(); render(END); }

  /* ── cross-highlighting: graph node <-> table row ──────────────────────── */
  function bindHighlight() {
    function set(i, on) {
      hovering[i] = on;
      if (rows[i]) rows[i].tr.classList.toggle("hi", on);
      if (gfx[i]) gfx[i].halo.style.opacity = on ? "0.85" : "0";
    }
    gfx.forEach(function (g, i) {
      g.hit.addEventListener("mouseenter", function () { set(i, true); });
      g.hit.addEventListener("mouseleave", function () { set(i, false); });
    });
    rows.forEach(function (r, i) {
      r.tr.addEventListener("mouseenter", function () { set(i, true); });
      r.tr.addEventListener("mouseleave", function () { set(i, false); });
    });
  }

  /* ── copy button ──────────────────────────────────────────────────────── */
  function flash(btn) {
    btn.textContent = "Copied";
    btn.classList.add("done");
    window.setTimeout(function () {
      btn.textContent = "Copy";
      btn.classList.remove("done");
    }, 1600);
  }

  function copyFallback(text, btn) {
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand("copy"); flash(btn); } catch (e) { /* leave the label alone */ }
    document.body.removeChild(ta);
  }

  function bindCopy() {
    var btn = document.getElementById("copy");
    if (!btn) return;
    btn.addEventListener("click", function () {
      var text = btn.getAttribute("data-copy") || "";
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(
          function () { flash(btn); },
          function () { copyFallback(text, btn); }
        );
      } else {
        copyFallback(text, btn);
      }
    });
  }

  /* ── init ─────────────────────────────────────────────────────────────── */
  buildGraph();
  buildLedger();
  bindHighlight();
  bindCopy();

  var replay = document.getElementById("replay");
  if (replay) {
    /* CSS already hides this under reduced-motion, but check at click time too:
       the control must never animate against a stated preference, however the
       user reached it. */
    replay.addEventListener("click", function () {
      if (reduced.matches) showFinal(); else play();
    });
  }

  if (reduced.matches) {
    showFinal();
  } else {
    render(0);
    /* Wait for the webfont so the counters don't reflow mid-count. */
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(play, play);
    } else {
      play();
    }
  }

  /* Respect the OS setting if it changes mid-visit. */
  function onReducedChange(e) { if (e.matches) showFinal(); }
  if (reduced.addEventListener) reduced.addEventListener("change", onReducedChange);
  else if (reduced.addListener) reduced.addListener(onReducedChange);
})();
