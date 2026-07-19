// MathJax config for pymdownx.arithmatex (generic mode).
// Must be set before the MathJax bundle loads.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

// MathJax is injected on the first page that actually contains math, never
// globally. Its own startup typesets that page; later instant-navigation
// swaps re-typeset here.
var mathjaxRequested = false;
document$.subscribe(function () {
  if (!document.querySelector(".arithmatex")) return;
  if (window.MathJax.typesetPromise) {
    MathJax.startup.output.clearCache();
    MathJax.typesetClear();
    MathJax.texReset();
    MathJax.typesetPromise();
  } else if (!mathjaxRequested) {
    mathjaxRequested = true;
    var script = document.createElement("script");
    script.src = "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js";
    document.head.appendChild(script);
  }
});
