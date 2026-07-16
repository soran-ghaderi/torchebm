// MathJax config for pymdownx.arithmatex (generic mode).
// Must execute before the deferred MathJax bundle loads.
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

// Re-typeset after each instant-navigation page swap. On the very first
// emission MathJax may not be ready yet; its own startup typesets that page.
document$.subscribe(function () {
  if (!window.MathJax || !MathJax.typesetPromise) return;
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
