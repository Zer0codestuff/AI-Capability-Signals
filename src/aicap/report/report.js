/* Interactive helpers: sortable/filterable tables + figure lightbox */

(function () {
  function textOf(el) {
    return (el.textContent || "").trim().toLowerCase();
  }

  function parseValue(raw) {
    const t = raw.replace(/,/g, "").trim();
    const n = Number(t);
    if (t !== "" && Number.isFinite(n)) return n;
    return t.toLowerCase();
  }

  function sortTable(table, colIndex, dir) {
    const tbody = table.tBodies[0];
    if (!tbody) return;
    const rows = Array.from(tbody.rows);
    rows.sort((a, b) => {
      const av = parseValue(a.cells[colIndex] ? a.cells[colIndex].textContent : "");
      const bv = parseValue(b.cells[colIndex] ? b.cells[colIndex].textContent : "");
      if (av < bv) return -1 * dir;
      if (av > bv) return 1 * dir;
      return 0;
    });
    rows.forEach((r) => tbody.appendChild(r));
  }

  document.querySelectorAll("table.sortable").forEach((table) => {
    const heads = table.querySelectorAll("thead th");
    heads.forEach((th, idx) => {
      let dir = 1;
      th.addEventListener("click", () => {
        sortTable(table, idx, dir);
        dir *= -1;
      });
    });
  });

  document.querySelectorAll(".table-wrap").forEach((wrap) => {
    const input = wrap.querySelector(".table-filter");
    const table = wrap.querySelector("table");
    if (!input || !table || !table.tBodies[0]) return;
    input.addEventListener("input", () => {
      const q = input.value.trim().toLowerCase();
      Array.from(table.tBodies[0].rows).forEach((row) => {
        row.style.display = !q || textOf(row).includes(q) ? "" : "none";
      });
    });
  });

  const lb = document.getElementById("lightbox");
  const lbImg = lb ? lb.querySelector("img") : null;
  const closeBtn = lb ? lb.querySelector(".lightbox-close") : null;

  function closeLightbox() {
    if (!lb) return;
    lb.hidden = true;
    if (lbImg) lbImg.removeAttribute("src");
  }

  document.querySelectorAll("figure.chart[data-lightbox] img").forEach((img) => {
    img.addEventListener("click", () => {
      if (!lb || !lbImg) return;
      lbImg.src = img.currentSrc || img.src;
      lbImg.alt = img.alt || "";
      lb.hidden = false;
    });
  });

  if (closeBtn) closeBtn.addEventListener("click", closeLightbox);
  if (lb) {
    lb.addEventListener("click", (e) => {
      if (e.target === lb) closeLightbox();
    });
  }
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeLightbox();
  });
})();
