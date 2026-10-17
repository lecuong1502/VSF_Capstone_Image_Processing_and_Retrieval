import { useCallback, useEffect, useRef, useState } from "react";
import { listQueues, loadQueue, saveDecision, type QueueSummary } from "./api";
import { onKey, start, type Item, type Session } from "./session";
import "./review.css";

const PREFETCH = 5;

export default function ReviewPage() {
  const queue = new URLSearchParams(window.location.search).get("queue");
  return queue ? <QueueReview name={queue} /> : <QueueList />;
}

function QueueList() {
  const [queues, setQueues] = useState<QueueSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    listQueues().then(setQueues, (e: Error) => setError(e.message));
  }, []);
  return (
    <main className="review-list">
      <h1>Kiểm duyệt thủ công</h1>
      {error && <p className="review-error">{error}</p>}
      {queues?.length === 0 && (
        <p>
          Chưa có hàng đợi. Tạo bằng{" "}
          <code>python -m pipeline.review queue relevance</code>.
        </p>
      )}
      <ul>
        {queues?.map((q) => (
          <li key={q.name}>
            <a href={`/review?queue=${encodeURIComponent(q.name)}`}>{q.name}</a>
            <span>
              {q.decided}/{q.total}
            </span>
            <progress value={q.decided} max={q.total || 1} />
          </li>
        ))}
      </ul>
    </main>
  );
}

function QueueReview({ name }: { name: string }) {
  const [session, setSession] = useState<Session | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saves, setSaves] = useState<number[]>([]); // save times in this visit, for items/hour
  // Key events can arrive faster than React re-renders: always step from the latest session.
  const current = useRef<Session | null>(null);
  const shownAt = useRef(0);

  const update = useCallback((next: Session | null) => {
    current.current = next;
    setSession(next);
  }, []);

  useEffect(() => {
    loadQueue(name).then(
      (q) => update(start(q.items, q.decisions)),
      (e: Error) => setError(e.message),
    );
  }, [name, update]);

  const index = session?.index;
  useEffect(() => {
    shownAt.current = performance.now();
  }, [index]);

  // Warm the browser cache so the next images appear instantly.
  useEffect(() => {
    if (!session) return;
    for (const item of session.items.slice(
      session.index + 1,
      session.index + 1 + PREFETCH,
    )) {
      for (const url of [item.image_url, ...item.reference_urls]) {
        if (url) new Image().src = url;
      }
    }
  }, [session]);

  const handleKey = useCallback(
    (event: KeyboardEvent) => {
      const before = current.current;
      if (!before || event.ctrlKey || event.metaKey || event.altKey) return;
      const result = onKey(before, event.key);
      if (result.session === before) return;
      event.preventDefault();
      update(result.session);
      const submission = result.submit;
      if (!submission) return;
      const now = performance.now();
      saveDecision(
        name,
        submission.item_id,
        submission.values,
        now - shownAt.current,
      ).then(
        () => {
          setSaves((times) => [...times, now]);
          setError(null);
        },
        (e: Error) => {
          // Keep the reviewer on the item that failed to save.
          setError(`Chưa lưu được ${submission.item_id}: ${e.message}`);
          const s = current.current;
          if (!s) return;
          const decisions = { ...s.decisions };
          delete decisions[submission.item_id];
          const back = s.items.findIndex(
            (i) => i.item_id === submission.item_id,
          );
          update({
            ...s,
            decisions,
            index: back,
            active: 0,
            values: submission.values,
          });
        },
      );
    },
    [name, update],
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [handleKey]);

  if (error && !session) return <p className="review-error">{error}</p>;
  if (!session) return <p className="review-status">Đang tải…</p>;

  const decided = Object.keys(session.decisions).length;
  const total = session.items.length;
  const span = saves.length >= 10 ? saves[saves.length - 1] - saves[0] : 0;
  const perHour =
    span > 0 ? Math.round(((saves.length - 1) / span) * 3_600_000) : null;
  const item = session.items[session.index];

  return (
    <main className="review">
      <header className="review-bar">
        <a href="/review">← Hàng đợi</a>
        <strong>{name}</strong>
        <span>
          {decided}/{total} đã quyết · mục {Math.min(session.index + 1, total)}
        </span>
        {perHour !== null && <span>{perHour} mục/giờ</span>}
        <progress value={decided} max={total || 1} />
      </header>
      {error && <p className="review-error">{error}</p>}
      {item ? (
        <ItemView item={item} session={session} />
      ) : (
        <section className="review-done">
          <h2>Xong hàng đợi 🎉</h2>
          <p>
            Xuất kết quả: <code>python -m pipeline.review export {name}</code>.
            Nhấn Backspace để xem lại.
          </p>
        </section>
      )}
      <footer className="review-keys">
        <kbd>phím lựa chọn</kbd> trả lời · <kbd>Enter</kbd> xác nhận ·{" "}
        <kbd>↑</kbd>
        <kbd>↓</kbd> đổi câu hỏi · <kbd>Backspace</kbd> mục trước · <kbd>→</kbd>{" "}
        bỏ qua
      </footer>
    </main>
  );
}

function ItemView({ item, session }: { item: Item; session: Session }) {
  return (
    <section className="review-item">
      <div className="review-media">
        {item.image_url ? (
          <img className="review-image" src={item.image_url} alt={item.title} />
        ) : (
          <div className="review-noimage">Không có ảnh</div>
        )}
      </div>
      <aside className="review-panel">
        <h2>{item.title}</h2>
        <dl className="review-context">
          {Object.entries(item.context).map(([k, v]) => (
            <div key={k}>
              <dt>{k}</dt>
              <dd>{v}</dd>
            </div>
          ))}
        </dl>
        {item.reference_urls.length > 0 && (
          <div className="review-refs" aria-label="Ảnh tham chiếu">
            {item.reference_urls.map((url) => (
              <img key={url} src={url} alt="" />
            ))}
          </div>
        )}
        <ol className="review-fields">
          {item.fields.map((field, i) => (
            <li
              key={field.name}
              className={i === session.active ? "active" : undefined}
            >
              <span className="review-field-label">
                {field.label || field.name}
              </span>
              <span className="review-options">
                {field.options.map((o) => (
                  <span
                    key={o.value}
                    className={
                      session.values[field.name] === o.value
                        ? "chosen"
                        : undefined
                    }
                  >
                    <kbd>{o.key}</kbd> {o.label || o.value}
                  </span>
                ))}
              </span>
            </li>
          ))}
        </ol>
      </aside>
    </section>
  );
}
