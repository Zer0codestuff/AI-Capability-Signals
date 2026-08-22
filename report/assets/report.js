
(function () {
  // Reading progress bar
  const bar = document.getElementById("progress-bar");
  if (bar) {
    const update = () => {
      const doc = document.documentElement;
      const max = doc.scrollHeight - doc.clientHeight;
      bar.style.width = max > 0 ? (doc.scrollTop / max) * 100 + "%" : "0%";
    };
    document.addEventListener("scroll", update, { passive: true });
    update();
  }

  // Active section chip in the top navigation
  const navLinks = Array.from(document.querySelectorAll(".topnav a"));
  if (navLinks.length && "IntersectionObserver" in window) {
    const byId = new Map(navLinks.map((a) => [a.getAttribute("href").slice(1), a]));
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          const link = byId.get(entry.target.id);
          if (!link) return;
          if (entry.isIntersecting) {
            navLinks.forEach((a) => a.classList.remove("active"));
            link.classList.add("active");
          }
        });
      },
      { rootMargin: "-20% 0px -70% 0px" }
    );
    document.querySelectorAll("section[id]").forEach((s) => observer.observe(s));
  }

  // Sortable + filterable tables
  document.querySelectorAll("table[data-sortable='true']").forEach((table) => {
    const headers = Array.from(table.querySelectorAll("th"));
    const wrap = table.closest(".table-card");
    const filter = wrap ? wrap.querySelector("[data-table-filter]") : null;
    const bodyRows = () => Array.from(table.querySelectorAll("tr")).slice(1);

    if (filter) {
      filter.addEventListener("input", () => {
        const q = filter.value.trim().toLowerCase();
        bodyRows().forEach((row) => {
          row.hidden = q.length > 0 && !row.textContent.toLowerCase().includes(q);
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

  // Figure lightbox
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
    const close = () => {
      lightbox.hidden = true;
      img.removeAttribute("src");
    };
    lightbox.querySelector("button").addEventListener("click", close);
    lightbox.addEventListener("click", (event) => {
      if (event.target === lightbox) close();
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !lightbox.hidden) close();
    });
  }
})();
