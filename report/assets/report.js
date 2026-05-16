
document.querySelectorAll("table[data-sortable='true']").forEach((table) => {
  const headers = Array.from(table.querySelectorAll("th"));
  const tableWrap = table.closest(".table-wrap");
  const filter = tableWrap?.querySelector("[data-table-filter]");
  const bodyRows = () => Array.from(table.querySelectorAll("tr")).slice(1);

  if (filter) {
    filter.addEventListener("input", () => {
      const query = filter.value.trim().toLowerCase();
      bodyRows().forEach((row) => {
        row.hidden = query.length > 0 && !row.textContent.toLowerCase().includes(query);
      });
    });
  }

  headers.forEach((header, index) => {
    header.addEventListener("click", () => {
      const rows = bodyRows();
      const direction = header.dataset.sortDir === "asc" ? "desc" : "asc";
      header.dataset.sortDir = direction;
      rows.sort((a, b) => {
        const av = a.children[index]?.textContent?.trim() || "";
        const bv = b.children[index]?.textContent?.trim() || "";
        const an = Number(av.replace(/[%,$]/g, ""));
        const bn = Number(bv.replace(/[%,$]/g, ""));
        const cmp = Number.isFinite(an) && Number.isFinite(bn) ? an - bn : av.localeCompare(bv);
        return direction === "asc" ? cmp : -cmp;
      });
      rows.forEach((row) => table.tBodies[0].appendChild(row));
    });
  });
});

const lightbox = document.getElementById("figure-lightbox");
if (lightbox) {
  const img = lightbox.querySelector("img");
  document.querySelectorAll("a[data-lightbox='figure']").forEach((link) => {
    link.addEventListener("click", (event) => {
      event.preventDefault();
      img.src = link.href;
      img.alt = link.querySelector("img")?.alt || "Expanded report figure";
      lightbox.hidden = false;
    });
  });
  lightbox.querySelector("button").addEventListener("click", () => {
    lightbox.hidden = true;
    img.removeAttribute("src");
  });
  lightbox.addEventListener("click", (event) => {
    if (event.target === lightbox) {
      lightbox.hidden = true;
      img.removeAttribute("src");
    }
  });
}
