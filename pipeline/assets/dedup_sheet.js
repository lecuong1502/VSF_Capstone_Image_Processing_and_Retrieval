// Keyboard labeling for reports/dedup_pairs.html. Labels are keyed by the image pair and
// stored per sheet, so regenerating the sheet never attaches old labels to different pairs.
const KEY = "dedup-labels:" + document.body.dataset.sheet;
let labels = {};
try {
  labels = JSON.parse(localStorage.getItem(KEY) || "{}");
} catch (e) {}
const cards = [...document.querySelectorAll(".pair")];
const pairKey = (c) => c.dataset.a + "-" + c.dataset.b;
let cur = 0;

function save() {
  try {
    localStorage.setItem(KEY, JSON.stringify(labels));
  } catch (e) {}
}

function render() {
  cards.forEach((c, i) => {
    c.classList.toggle("current", i === cur);
    c.dataset.label = labels[pairKey(c)] || "";
  });
  document.getElementById("done").textContent = Object.keys(labels).length;
  if (cards[cur]) cards[cur].scrollIntoView({ block: "center" });
}

function setLabel(value) {
  labels[pairKey(cards[cur])] = value;
  save();
  cur = Math.min(cur + 1, cards.length - 1);
  render();
}

document.addEventListener("keydown", (e) => {
  if (!cards.length || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key === "y") setLabel("yes");
  else if (e.key === "n") setLabel("no");
  else if (e.key === "s") setLabel("skip");
  else if (e.key === "ArrowDown" || e.key === "j") cur = Math.min(cur + 1, cards.length - 1);
  else if (e.key === "ArrowUp" || e.key === "k") cur = Math.max(cur - 1, 0);
  else return;
  e.preventDefault();
  render();
});

cards.forEach((c, i) =>
  c.addEventListener("click", () => {
    cur = i;
    render();
  }),
);

document.getElementById("export").addEventListener("click", () => {
  const rows = ["image_a,image_b,distance,label"];
  cards.forEach((c) => {
    const v = labels[pairKey(c)];
    if (v) rows.push([c.dataset.a, c.dataset.b, c.dataset.d, v].join(","));
  });
  const url = URL.createObjectURL(new Blob([rows.join("\n") + "\n"], { type: "text/csv" }));
  const a = Object.assign(document.createElement("a"), { href: url, download: "dedup_labels.csv" });
  a.click();
  URL.revokeObjectURL(url);
});

render();
