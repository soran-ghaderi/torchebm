/**
 * TorchEBM Interactive Benchmark Dashboard
 * Activates only on pages with #bench-app; dynamically loads Plotly.js.
 * Designed to run inside MkDocs Material — adapts to light/dark theme.
 */
(function () {
  "use strict";

  /* ── Bootstrap ──────────────────────────────────────────────── */
  function boot() {
    var app = document.getElementById("bench-app");
    if (!app || app.dataset.init) return;
    app.dataset.init = "1";
    var d = window.__BENCH_DATA__;
    if (!d) return;
    if (window.Plotly) {
      initDashboard(d);
      return;
    }
    var s = document.createElement("script");
    s.src = "https://cdn.plot.ly/plotly-2.32.0.min.js";
    s.onload = function () {
      initDashboard(d);
    };
    document.head.appendChild(s);
  }

  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", boot);
  else boot();

  /* ── Constants ──────────────────────────────────────────────── */
  var SCALE_ORDER = { small: 0, medium: 1, large: 2 };
  var SCALE_COLORS = { small: "#58a6ff", medium: "#d29922", large: "#f85149" };
  var ALL_BENCHMARKS, RUN_META, MODULES, SCALES;

  /* ── Theme helpers ──────────────────────────────────────────── */
  function isDark() {
    return document.querySelector('[data-md-color-scheme="slate"]') !== null;
  }

  function baseLayout(ov) {
    var dk = isDark(),
      fg = dk ? "#c9d1d9" : "#24292f",
      g = dk ? "#21262d" : "#e0e0e0";
    ov = ov || {};
    return {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: dk ? "rgba(255,255,255,0.02)" : "rgba(0,0,0,0.015)",
      font: { color: fg, size: 12 },
      xaxis: Object.assign({ gridcolor: g, zeroline: false, automargin: true }, ov.xaxis || {}),
      yaxis: Object.assign({ gridcolor: g, zeroline: false, automargin: true }, ov.yaxis || {}),
      legend: Object.assign(
        { bgcolor: "rgba(0,0,0,0)", font: { size: 13, color: fg } },
        ov.legend || {}
      ),
      margin: ov.margin || { t: 36, b: 70, l: 70, r: 20 },
      hovermode: "closest",
      bargap: 0.15,
      bargroupgap: 0.1,
      barmode: ov.barmode || undefined,
      height: ov.height || 380,
      showlegend: ov.showlegend !== undefined ? ov.showlegend : undefined,
      shapes: ov.shapes || undefined,
    };
  }

  /* ── Utility functions ──────────────────────────────────────── */
  function getRunBenches(idx) {
    return ALL_BENCHMARKS.filter(function (b) {
      return b.run_index === parseInt(idx);
    });
  }

  function benchBase(sn) {
    var base = sn.replace("test_", "");
    if (base.indexOf("[") > -1) {
      var parts = base.split("["),
        name = parts[0],
        ps = parts[1].replace("]", "");
      var extras = ps
        .split("-")
        .filter(function (p) {
          return p.indexOf("scale=") !== 0;
        })
        .map(function (p) {
          return p.indexOf("=") > -1 ? p.split("=")[1] : p;
        })
        .filter(Boolean);
      return extras.length ? name + " (" + extras.join(", ") + ")" : name;
    }
    return base;
  }

  function autoUnit(ms) {
    if (ms >= 1000) return (ms / 1000).toFixed(2) + " s";
    if (ms >= 1) return ms.toFixed(3) + " ms";
    return (ms * 1000).toFixed(1) + " \u00b5s";
  }

  function pillHtml(sp) {
    var cls =
      sp >= 1.05
        ? "bench-pill-green"
        : sp < 0.95
          ? "bench-pill-red"
          : sp >= 1.01
            ? "bench-pill-yellow"
            : "bench-pill-neutral";
    return (
      '<span class="bench-pill ' + cls + '">' + sp.toFixed(2) + "x</span>"
    );
  }

  function deltaBar(pct, maxPct) {
    var w = Math.min((Math.abs(pct) / (maxPct || 20)) * 80, 80);
    var cls = pct >= 0 ? "bench-delta-positive" : "bench-delta-negative";
    return (
      '<span class="bench-delta-bar ' +
      cls +
      '" style="width:' +
      Math.max(w, 4) +
      'px"></span> ' +
      (pct >= 0 ? "+" : "") +
      pct.toFixed(1) +
      "%"
    );
  }

  function geomean(vals) {
    var v = vals.filter(function (x) {
      return x > 0 && isFinite(x);
    });
    if (!v.length) return 1;
    return Math.exp(
      v.reduce(function (s, x) {
        return s + Math.log(x);
      }, 0) / v.length
    );
  }

  function fN(v, d) {
    return v != null ? v.toFixed(d) : "\u2014";
  }
  function fI(v) {
    return v != null ? v.toLocaleString() : "\u2014";
  }
  function fM(v) {
    return v != null ? v.toFixed(1) : "\u2014";
  }
  function fP(v) {
    return v != null ? v.toFixed(2) + "M" : "\u2014";
  }

  function unique(arr) {
    var s = {};
    return arr.filter(function (x) {
      return s[x] ? false : (s[x] = true);
    });
  }

  function makeBarTraces(chartData, bases, scaleProp, valFn, hoverFmt) {
    var traces = [];
    SCALES.forEach(function (scale) {
      var sd = chartData.filter(function (b) {
        return b.scale === scale;
      });
      if (!sd.length) return;
      var xs = [],
        ys = [],
        errs = [];
      bases.forEach(function (base) {
        var m = sd.find(function (b) {
          return benchBase(b.short_name) === base;
        });
        if (m) {
          xs.push(base);
          ys.push(valFn(m));
          if (m.stddev_ms !== undefined) errs.push(m.stddev_ms);
        }
      });
      var trace = {
        x: xs,
        y: ys,
        name: scale,
        type: "bar",
        marker: { color: SCALE_COLORS[scale] },
        hovertemplate: hoverFmt,
      };
      traces.push(trace);
    });
    return traces;
  }

  /* ── Tab switching ──────────────────────────────────────────── */
  var renderers = {
    overview: renderOverview,
    comparison: renderComparison,
    scaling: renderScaling,
    history: renderHistory,
    details: renderDetails,
  };

  window.benchSwitchTab = function (name) {
    document.querySelectorAll(".bench-nav-tab").forEach(function (t) {
      t.classList.remove("bench-active");
    });
    document.querySelectorAll(".bench-tab-content").forEach(function (t) {
      t.classList.remove("bench-active");
    });
    var btn = document.querySelector('.bench-nav-tab[data-tab="' + name + '"]');
    if (btn) btn.classList.add("bench-active");
    var tab = document.getElementById("bench-tab-" + name);
    if (tab) tab.classList.add("bench-active");
    if (renderers[name]) renderers[name]();
  };

  /* ── Global render triggers for select change handlers ──── */
  window.benchRenderOverview = renderOverview;
  window.benchRenderComparison = renderComparison;
  window.benchRenderScaling = renderScaling;
  window.benchRenderHistory = renderHistory;
  window.benchRenderDetails = renderDetails;

  /* ── Init ───────────────────────────────────────────────────── */
  function initDashboard(data) {
    ALL_BENCHMARKS = data.benchmarks;
    RUN_META = data.run_meta;
    MODULES = data.modules;
    SCALES = data.scales;
    renderOverview();

    // Re-render on theme change
    new MutationObserver(function () {
      var active = document.querySelector(".bench-nav-tab.bench-active");
      if (active && renderers[active.dataset.tab]) renderers[active.dataset.tab]();
    }).observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-md-color-scheme"],
    });
  }

  /* ═══════════════════════════════════════════════════════════════
     OVERVIEW — per-module sections with charts + table
     ═══════════════════════════════════════════════════════════════ */
  function buildSummaryCards(mod, d) {
    var c = "";
    var hasMem = d.some(function (b) {
      return b.peak_memory_mb != null;
    });
    if (hasMem) {
      var ms = d
        .filter(function (b) {
          return b.peak_memory_mb != null;
        })
        .map(function (b) {
          return b.peak_memory_mb;
        });
      c +=
        '<div class="bench-mc"><span class="bench-mc-l">Peak Memory</span><span class="bench-mc-v">' +
        Math.min.apply(null, ms).toFixed(1) +
        " \u2013 " +
        Math.max.apply(null, ms).toFixed(1) +
        " MB</span></div>";
    }
    if (mod === "samplers") {
      var ev = d.filter(function (b) {
        return b.ess_per_sec != null;
      });
      if (ev.length) {
        var avg =
          ev.reduce(function (s, b) {
            return s + b.ess_per_sec;
          }, 0) / ev.length;
        c +=
          '<div class="bench-mc"><span class="bench-mc-l">Avg ESS/sec</span><span class="bench-mc-v">' +
          avg.toFixed(1) +
          "</span></div>";
      }
      var av = d.filter(function (b) {
        return b.acceptance_rate != null;
      });
      if (av.length) {
        var best = Math.max.apply(
          null,
          av.map(function (b) {
            return b.acceptance_rate;
          })
        );
        c +=
          '<div class="bench-mc"><span class="bench-mc-l">Best Acc. Rate</span><span class="bench-mc-v bench-ok">' +
          (best * 100).toFixed(1) +
          "%</span></div>";
      }
    }
    if (mod === "losses") {
      var fv = d.filter(function (b) {
        return b.loss_is_finite != null;
      });
      if (fv.length) {
        var ok = fv.every(function (b) {
          return b.loss_is_finite;
        });
        c +=
          '<div class="bench-mc"><span class="bench-mc-l">Losses Finite</span><span class="bench-mc-v ' +
          (ok ? "bench-ok" : "bench-warn") +
          '">' +
          (ok ? "All OK" : "Some NaN/Inf") +
          "</span></div>";
      }
    }
    if (mod === "models") {
      var fl = d.filter(function (b) {
        return b.gflops_per_sec != null;
      });
      if (fl.length) {
        var pk = Math.max.apply(
          null,
          fl.map(function (b) {
            return b.gflops_per_sec;
          })
        );
        c +=
          '<div class="bench-mc"><span class="bench-mc-l">Peak GFLOPS/s</span><span class="bench-mc-v">' +
          pk.toFixed(1) +
          "</span></div>";
      }
    }
    return c ? '<div class="bench-mcs">' + c + "</div>" : "";
  }

  function renderOverview() {
    var runIdx = parseInt(
      document.getElementById("bench-ov-run").value
    );
    var data = getRunBenches(runIdx);
    var ct = document.getElementById("bench-overview-container");
    ct.innerHTML = "";

    MODULES.forEach(function (mod) {
      var modData = data.filter(function (b) {
        return b.module === mod;
      });
      if (!modData.length) return;
      var eagerData = modData.filter(function (b) {
        return b.bench_mode === "eager";
      });
      var chartData = eagerData.length ? eagerData : modData;
      var bases = unique(
        chartData.map(function (b) {
          return benchBase(b.short_name);
        })
      );

      var cId = "bench-ov-t-" + mod,
        oId = "bench-ov-o-" + mod,
        mId = "bench-ov-m-" + mod;
      var hasMem = chartData.some(function (b) {
        return b.peak_memory_mb != null;
      });
      var cards = buildSummaryCards(mod, modData);

      var sorted = modData.slice().sort(function (a, b) {
        return (
          benchBase(a.short_name).localeCompare(benchBase(b.short_name)) ||
          (SCALE_ORDER[a.scale] || 0) - (SCALE_ORDER[b.scale] || 0)
        );
      });
      var tbody = sorted
        .map(function (b) {
          return (
            "<tr><td>" +
            benchBase(b.short_name) +
            '</td><td><span class="bench-tag">' +
            b.scale +
            "</span></td><td><b>" +
            autoUnit(b.median_ms) +
            "</b></td><td>" +
            autoUnit(b.mean_ms) +
            "</td><td>" +
            autoUnit(b.stddev_ms) +
            "</td><td>" +
            b.ops.toFixed(1) +
            "</td><td>" +
            fN(b.samples_per_sec, 1) +
            "</td><td>" +
            fM(b.peak_memory_mb) +
            "</td><td>" +
            b.rounds +
            "</td></tr>"
          );
        })
        .join("");

      var sec = document.createElement("div");
      sec.className = "bench-mod-section";
      var chartsH =
        '<div class="bench-grid-2"><div class="bench-card"><h3>Median Time by Scale</h3><div id="' +
        cId +
        '" class="bench-chart"></div></div>' +
        '<div class="bench-card"><h3>Throughput (ops/sec)</h3><div id="' +
        oId +
        '" class="bench-chart"></div></div></div>';
      if (hasMem)
        chartsH +=
          '<div class="bench-grid-2"><div class="bench-card"><h3>Peak Memory (MB)</h3><div id="' +
          mId +
          '" class="bench-chart"></div></div><div></div></div>';

      sec.innerHTML =
        '<div class="bench-mod-hdr">' +
        mod +
        ' <span style="font-size:0.7em;opacity:0.6">(' +
        modData.length +
        " benchmarks)</span></div>" +
        cards +
        chartsH +
        '<div class="bench-card"><h3>Stats</h3><div class="bench-tbl-scroll"><table class="bench-tbl">' +
        "<thead><tr><th>Benchmark</th><th>Scale</th><th>Median</th><th>Mean</th><th>Std Dev</th><th>Ops/s</th><th>Samp/s</th><th>Mem (MB)</th><th>Rounds</th></tr></thead>" +
        "<tbody>" +
        tbody +
        "</tbody></table></div></div>";
      ct.appendChild(sec);

      // Time bars
      var barH = Math.max(420, bases.length * 28 + 140);
      Plotly.newPlot(
        cId,
        makeBarTraces(
          chartData,
          bases,
          "scale",
          function (b) {
            return b.median_ms;
          },
          "%{x}<br>%{y:.4f} ms<extra></extra>"
        ),
        baseLayout({
          barmode: "group",
          height: barH,
          xaxis: { tickangle: -75 },
          yaxis: { title: "Median Time (ms)" },
          legend: { orientation: "h", y: 1.12 },
        }),
        { responsive: true }
      );

      // Ops bars
      Plotly.newPlot(
        oId,
        makeBarTraces(
          chartData,
          bases,
          "scale",
          function (b) {
            return b.ops;
          },
          "%{x}<br>%{y:.1f} ops/s<extra></extra>"
        ),
        baseLayout({
          barmode: "group",
          height: barH,
          xaxis: { tickangle: -45 },
          yaxis: { title: "Operations / sec" },
          legend: { orientation: "h", y: 1.12 },
        }),
        { responsive: true }
      );

      // Memory bars
      if (hasMem) {
        var memD = chartData.filter(function (b) {
          return b.peak_memory_mb != null;
        });
        var memBases = bases.filter(function (base) {
              return memD.some(function (b) {
                return benchBase(b.short_name) === base;
              });
            });
        var memH = Math.max(420, memBases.length * 28 + 140);
        Plotly.newPlot(
          mId,
          makeBarTraces(
            memD,
            memBases,
            "scale",
            function (b) {
              return b.peak_memory_mb;
            },
            "%{x}<br>%{y:.2f} MB<extra></extra>"
          ),
          baseLayout({
            barmode: "group",
            height: memH,
            xaxis: { tickangle: -45 },
            yaxis: { title: "Peak Memory (MB)" },
            legend: { orientation: "h", y: 1.12 },
          }),
          { responsive: true }
        );
      }
    });
  }

  /* ═══════════════════════════════════════════════════════════════
     COMPARISON — per-module speedup charts and tables
     ═══════════════════════════════════════════════════════════════ */
  function renderComparison() {
    var bIdx = parseInt(document.getElementById("bench-cmp-before").value);
    var aIdx = parseInt(document.getElementById("bench-cmp-after").value);
    var bData = getRunBenches(bIdx),
      aData = getRunBenches(aIdx);
    var bMap = {};
    bData.forEach(function (b) {
      bMap[b.fullname] = b;
    });

    var compsByMod = {};
    aData.forEach(function (b) {
      if (!bMap[b.fullname]) return;
      var bb = bMap[b.fullname];
      var sp = bb.median_ms / b.median_ms;
      var comp = {
        short_name: b.short_name,
        module: b.module,
        scale: b.scale,
        before_ms: bb.median_ms,
        after_ms: b.median_ms,
        speedup: sp,
        pct: (1 - b.median_ms / bb.median_ms) * 100,
      };
      if (!compsByMod[b.module]) compsByMod[b.module] = [];
      compsByMod[b.module].push(comp);
    });

    var sumEl = document.getElementById("bench-cmp-summary");
    var ct = document.getElementById("bench-cmp-modules");
    var all = [];
    Object.keys(compsByMod).forEach(function (m) {
      all = all.concat(compsByMod[m]);
    });

    if (!all.length) {
      sumEl.innerHTML =
        '<div class="bench-card" style="grid-column:1/-1"><p style="opacity:0.6">No overlapping benchmarks between selected runs.</p></div>';
      ct.innerHTML = "";
      return;
    }

    var improved = all.filter(function (c) {
        return c.pct > 1;
      }).length,
      regressed = all.filter(function (c) {
        return c.pct < -1;
      }).length;
    var modGm = Object.keys(compsByMod)
      .sort()
      .map(function (mod) {
        var gm = geomean(
          compsByMod[mod].map(function (c) {
            return c.speedup;
          })
        );
        var clr = gm >= 1 ? "var(--bench-green)" : "var(--bench-red)";
        return (
          '<div class="bench-sum-card"><div class="bench-sum-label">' +
          mod +
          ' geomean</div><div class="bench-sum-value" style="color:' +
          clr +
          '">' +
          gm.toFixed(3) +
          "x</div><div class=\"bench-sum-detail\">" +
          compsByMod[mod].length +
          " benchmarks</div></div>"
        );
      })
      .join("");

    sumEl.innerHTML =
      '<div class="bench-sum-card"><div class="bench-sum-label">Total Compared</div><div class="bench-sum-value">' +
      all.length +
      "</div></div>" +
      '<div class="bench-sum-card"><div class="bench-sum-label">Improved (>1%)</div><div class="bench-sum-value" style="color:var(--bench-green)">' +
      improved +
      "</div></div>" +
      '<div class="bench-sum-card"><div class="bench-sum-label">Regressed (>1%)</div><div class="bench-sum-value" style="color:var(--bench-red)">' +
      regressed +
      "</div></div>" +
      modGm;

    ct.innerHTML = "";
    MODULES.forEach(function (mod) {
      var comps = compsByMod[mod];
      if (!comps || !comps.length) return;
      comps.sort(function (a, b) {
        return a.pct - b.pct;
      });
      var maxPct = Math.max.apply(
        null,
        comps
          .map(function (c) {
            return Math.abs(c.pct);
          })
          .concat([1])
      );
      var chId = "bench-cmp-" + mod;

      var sec = document.createElement("div");
      sec.className = "bench-mod-section";
      sec.innerHTML =
        '<div class="bench-mod-hdr">' +
        mod +
        "</div>" +
        '<div class="bench-card"><div id="' +
        chId +
        '" class="bench-chart"></div></div>' +
        '<div class="bench-card"><div class="bench-tbl-scroll"><table class="bench-tbl">' +
        "<thead><tr><th>Benchmark</th><th>Scale</th><th>Before (ms)</th><th>After (ms)</th><th>Speedup</th><th>Change</th><th>Delta</th></tr></thead><tbody>" +
        comps
          .map(function (c) {
            return (
              "<tr><td>" +
              benchBase(c.short_name) +
              '</td><td><span class="bench-tag">' +
              c.scale +
              "</span></td><td>" +
              c.before_ms.toFixed(4) +
              "</td><td>" +
              c.after_ms.toFixed(4) +
              "</td><td>" +
              pillHtml(c.speedup) +
              '</td><td style="color:' +
              (c.pct >= 0 ? "var(--bench-green)" : "var(--bench-red)") +
              '">' +
              (c.pct >= 0 ? "+" : "") +
              c.pct.toFixed(2) +
              "%</td><td>" +
              deltaBar(c.pct, maxPct) +
              "</td></tr>"
            );
          })
          .join("") +
        "</tbody></table></div></div>";
      ct.appendChild(sec);

      // Horizontal bar chart
      var barColors = comps.map(function (c) {
        return c.pct >= 0 ? "#3fb950" : "#f85149";
      });
      Plotly.newPlot(
        chId,
        [
          {
            y: comps.map(function (c) {
              return benchBase(c.short_name) + " [" + c.scale + "]";
            }),
            x: comps.map(function (c) {
              return c.pct;
            }),
            type: "bar",
            orientation: "h",
            marker: { color: barColors },
            customdata: comps.map(function (c) {
              return [c.before_ms, c.after_ms, c.speedup];
            }),
            hovertemplate:
              "%{y}<br>Change: %{x:+.2f}%<br>Before: %{customdata[0]:.4f} ms<br>After: %{customdata[1]:.4f} ms<br>Speedup: %{customdata[2]:.3f}x<extra></extra>",
          },
        ],
        baseLayout({
          showlegend: false,
          height: Math.max(250, comps.length * 32 + 60),
          xaxis: {
            title: "Change (%) \u2014 positive = faster",
            zeroline: true,
            zerolinecolor: "#484f58",
            zerolinewidth: 2,
          },
          yaxis: { automargin: true },
        }),
        { responsive: true }
      );
    });
  }

  /* ═══════════════════════════════════════════════════════════════
     SCALING — absolute & relative scaling curves per module
     ═══════════════════════════════════════════════════════════════ */
  function renderScaling() {
    var mod = document.getElementById("bench-sc-module").value;
    var runIdx = parseInt(document.getElementById("bench-sc-run").value);
    var data = getRunBenches(runIdx).filter(function (b) {
      return b.module === mod;
    });
    var bases = unique(
      data.map(function (b) {
        return benchBase(b.short_name);
      })
    );

    // Absolute
    var absTraces = bases.map(function (base) {
      var pts = SCALES.map(function (s) {
        return data.find(function (b) {
          return benchBase(b.short_name) === base && b.scale === s;
        });
      }).filter(Boolean);
      return {
        x: pts.map(function (p) {
          return p.scale;
        }),
        y: pts.map(function (p) {
          return p.median_ms;
        }),
        error_y: {
          type: "data",
          array: pts.map(function (p) {
            return p.stddev_ms;
          }),
          visible: true,
          thickness: 1.5,
        },
        name: base,
        type: "scatter",
        mode: "lines+markers",
        marker: { size: 8 },
        hovertemplate: base + "<br>%{x}: %{y:.4f} ms<extra></extra>",
      };
    });
    var scaleH = Math.max(450, bases.length * 20 + 150);
    Plotly.react(
      "bench-sc-abs",
      absTraces,
      baseLayout({
        height: scaleH,
        xaxis: {
          title: "Scale",
          categoryorder: "array",
          categoryarray: ["small", "medium", "large"],
        },
        yaxis: { title: "Median Time (ms)" },
      }),
      { responsive: true }
    );

    // Relative
    var relTraces = bases
      .map(function (base) {
        var sm = data.find(function (b) {
          return benchBase(b.short_name) === base && b.scale === "small";
        });
        if (!sm || sm.median_ms === 0) return null;
        var pts = SCALES.map(function (s) {
          return data.find(function (b) {
            return benchBase(b.short_name) === base && b.scale === s;
          });
        }).filter(Boolean);
        return {
          x: pts.map(function (p) {
            return p.scale;
          }),
          y: pts.map(function (p) {
            return p.median_ms / sm.median_ms;
          }),
          name: base,
          type: "scatter",
          mode: "lines+markers",
          marker: { size: 8 },
          hovertemplate:
            base + "<br>%{x}: %{y:.2f}x vs small<extra></extra>",
        };
      })
      .filter(Boolean);
    Plotly.react(
      "bench-sc-rel",
      relTraces,
      baseLayout({
        height: scaleH,
        xaxis: {
          title: "Scale",
          categoryorder: "array",
          categoryarray: ["small", "medium", "large"],
        },
        yaxis: { title: "Multiplier (1x = small)" },
      }),
      { responsive: true }
    );

    // Table
    var tb = document.querySelector("#bench-sc-table tbody");
    if (tb)
      tb.innerHTML = bases
        .map(function (base) {
          var sm = data.find(function (b) {
            return benchBase(b.short_name) === base && b.scale === "small";
          });
          var md = data.find(function (b) {
            return benchBase(b.short_name) === base && b.scale === "medium";
          });
          var lg = data.find(function (b) {
            return benchBase(b.short_name) === base && b.scale === "large";
          });
          return (
            "<tr><td>" +
            base +
            "</td><td>" +
            (sm ? autoUnit(sm.median_ms) : "\u2014") +
            "</td><td>" +
            (md ? autoUnit(md.median_ms) : "\u2014") +
            "</td><td>" +
            (lg ? autoUnit(lg.median_ms) : "\u2014") +
            "</td><td>" +
            (sm && md ? (md.median_ms / sm.median_ms).toFixed(2) + "x" : "\u2014") +
            "</td><td>" +
            (sm && lg ? (lg.median_ms / sm.median_ms).toFixed(2) + "x" : "\u2014") +
            "</td></tr>"
          );
        })
        .join("");
  }

  /* ═══════════════════════════════════════════════════════════════
     HISTORY — time series per benchmark across runs
     ═══════════════════════════════════════════════════════════════ */
  function renderHistory() {
    var mod = document.getElementById("bench-hist-module").value;
    var sf = document.getElementById("bench-hist-scale").value;
    var data = ALL_BENCHMARKS.filter(function (b) {
      return b.module === mod;
    });
    if (sf !== "all")
      data = data.filter(function (b) {
        return b.scale === sf;
      });

    var title = document.getElementById("bench-hist-title");
    if (title)
      title.textContent =
        mod +
        " \u2014 Performance Over Time" +
        (sf !== "all" ? " (" + sf + ")" : "");

    var byName = {};
    data.forEach(function (b) {
      var key =
        benchBase(b.short_name) +
        (sf === "all" ? " [" + b.scale + "]" : "");
      if (!byName[key]) byName[key] = [];
      byName[key].push(b);
    });

    var traces = Object.keys(byName).map(function (name) {
      var pts = byName[name].sort(function (a, b) {
        return a.run_datetime.localeCompare(b.run_datetime);
      });
      return {
        x: pts.map(function (p) {
          return p.run_label;
        }),
        y: pts.map(function (p) {
          return p.median_ms;
        }),
        error_y: {
          type: "data",
          array: pts.map(function (p) {
            return p.stddev_ms;
          }),
          visible: true,
          thickness: 1,
        },
        name: name,
        type: "scatter",
        mode: "lines+markers",
        marker: { size: 7 },
        hovertemplate: name + "<br>%{y:.4f} ms<br>%{x}<extra></extra>",
      };
    });

    Plotly.react(
      "bench-hist-chart",
      traces,
      baseLayout({
        height: 550,
        xaxis: { title: "Run", tickangle: -30 },
        yaxis: { title: "Median Time (ms)" },
      }),
      { responsive: true }
    );
  }

  /* ═══════════════════════════════════════════════════════════════
     DETAILS — full searchable table grouped by module
     ═══════════════════════════════════════════════════════════════ */
  function renderDetails() {
    var runIdx = parseInt(document.getElementById("bench-dt-run").value);
    var modF = document.getElementById("bench-dt-module").value;
    var search = (document.getElementById("bench-dt-search").value || "").toLowerCase();
    var ct = document.getElementById("bench-details-container");

    var data = getRunBenches(runIdx);
    if (modF !== "all")
      data = data.filter(function (b) {
        return b.module === modF;
      });
    if (search)
      data = data.filter(function (b) {
        return (
          b.short_name.toLowerCase().indexOf(search) > -1 ||
          b.module.toLowerCase().indexOf(search) > -1
        );
      });

    var mods =
      modF !== "all"
        ? [modF]
        : MODULES.filter(function (m) {
            return data.some(function (b) {
              return b.module === m;
            });
          });

    ct.innerHTML = mods
      .map(function (mod) {
        var md = data
          .filter(function (b) {
            return b.module === mod;
          })
          .sort(function (a, b) {
            return (
              benchBase(a.short_name).localeCompare(benchBase(b.short_name)) ||
              (SCALE_ORDER[a.scale] || 0) - (SCALE_ORDER[b.scale] || 0)
            );
          });
        if (!md.length) return "";
        return (
          '<div class="bench-mod-section"><div class="bench-mod-hdr">' +
          mod +
          ' <span style="font-size:0.7em;opacity:0.6">(' +
          md.length +
          " benchmarks)</span></div>" +
          '<div class="bench-card"><div class="bench-tbl-scroll"><table class="bench-tbl">' +
          "<thead><tr><th>Benchmark</th><th>Scale</th><th>Median</th><th>Mean</th><th>Std Dev</th><th>Min</th><th>Max</th><th>IQR</th>" +
          "<th>Ops/s</th><th>Samp/s</th><th>GFLOPS/s</th><th>Mem (MB)</th><th>ESS</th><th>ESS/s</th><th>Acc. Rate</th><th>Loss</th><th>Rounds</th><th>Outliers</th></tr></thead><tbody>" +
          md
            .map(function (b) {
              return (
                "<tr><td>" +
                benchBase(b.short_name) +
                '</td><td><span class="bench-tag">' +
                b.scale +
                "</span></td><td><b>" +
                autoUnit(b.median_ms) +
                "</b></td><td>" +
                autoUnit(b.mean_ms) +
                "</td><td>" +
                autoUnit(b.stddev_ms) +
                "</td><td>" +
                autoUnit(b.min_ms) +
                "</td><td>" +
                autoUnit(b.max_ms) +
                "</td><td>" +
                autoUnit(b.iqr_ms) +
                "</td><td>" +
                b.ops.toFixed(1) +
                "</td><td>" +
                fN(b.samples_per_sec, 1) +
                "</td><td>" +
                fN(b.gflops_per_sec, 1) +
                "</td><td>" +
                fM(b.peak_memory_mb) +
                "</td><td>" +
                fN(b.ess, 1) +
                "</td><td>" +
                fN(b.ess_per_sec, 1) +
                "</td><td>" +
                (b.acceptance_rate != null
                  ? (b.acceptance_rate * 100).toFixed(1) + "%"
                  : "\u2014") +
                "</td><td>" +
                (b.loss_value != null
                  ? b.loss_is_finite
                    ? b.loss_value.toFixed(4)
                    : '<span style="color:var(--bench-red)">NaN/Inf</span>'
                  : "\u2014") +
                "</td><td>" +
                b.rounds +
                "</td><td>" +
                (b.outliers || "\u2014") +
                "</td></tr>"
              );
            })
            .join("") +
          "</tbody></table></div></div></div>"
        );
      })
      .join("");
  }
})();
