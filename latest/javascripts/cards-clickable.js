// Make Material grid cards fully clickable by overlaying the first link
// Works for lists rendered by `.grid.cards` on any page
(function () {
  function enhanceCards(root) {
    if (!root) return;
    var cardLists = root.querySelectorAll('.md-typeset .grid.cards > ul, .md-typeset .grid.cards > ol');
    cardLists.forEach(function (list) {
      list.querySelectorAll(':scope > li').forEach(function (item) {
        if (item.classList.contains('card-clickable')) return;
        var firstLink = item.querySelector('a[href]');
        if (!firstLink) return;
        var href = firstLink.getAttribute('href');
        if (!href) return;

        // mark clickable and add overlay anchor
        item.classList.add('card-clickable');
        var overlay = document.createElement('a');
        overlay.className = 'card-link-overlay';
        overlay.href = href;
        overlay.setAttribute('aria-label', firstLink.textContent || 'Open');
        overlay.tabIndex = 0;
        item.appendChild(overlay);
      });
    });
  }

  // Run on initial load
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () { enhanceCards(document); });
  } else {
    enhanceCards(document);
  }

  // Run again on navigation (Material SPA behavior)
  document.addEventListener('DOMContentSwitch', function () { enhanceCards(document); });
})();


