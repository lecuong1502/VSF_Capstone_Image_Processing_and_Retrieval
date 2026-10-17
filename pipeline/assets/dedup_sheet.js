const KEY = "dedup-labels-v1";
let labels = {};
try { labels = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
const cards = [...document.querySelectorAll(".pair")];
let cur = 0;
function save() { try { localStorage.setItem(KEY, JSON.stringify(labels)); } catch (e) {} }
function render() {
  cards.forEach((c, i) => {
    c.classList.toggle("current", i === cur);
    const v = labels[c.dataset.id];
    c.dataset.label = v === undefined ? "" : v;
  });
  document.getElementById("done").textContent = Object.keys(labels).length;
  cards[cur].scrollIntoView({block: "center"});
}
function setLabel(v) { labels[cards[cur].dataset.id] = v; save(); cur = Math.min(cur + 1, cards.length - 1); render(); }
document.addEventListener("keydown", e => {
  if (e.key === "y") setLabel("yes");
  else if (e.key === "n") setLabel("no");
  else if (e.key === "s") setLabel("skip");
  else if (e.key === "ArrowDown" || e.key === "j") { cur = Math.min(cur + 1, cards.length - 1); render(); }
  else if (e.key === "ArrowUp" || e.key === "k") { cur = Math.max(cur - 1, 0); render(); }
  else return;
  e.preventDefault();
});
cards.forEach((c, i) => c.addEventListener("click", () => { cur = i; render(); }));
document.getElementById("export").addEventListener("click", () => {
  const rows = ["pair_id,image_a,image_b,distance,label"];
  cards.forEach(c => { const v = labels[c.dataset.id]; if (v) rows.push([c.dataset.id, c.dataset.a, c.dataset.b, c.dataset.d, v].join(",")); });
  const url = URL.createObjectURL(new Blob([rows.join("\n") + "\n"], {type: "text/csv"}));
  const a = Object.assign(document.createElement("a"), {href: url, download: "dedup_labels.csv"});
  a.click(); URL.revokeObjectURL(url);
});
render();
