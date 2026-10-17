import type { Item, Values } from "./session";

export type QueueSummary = { name: string; total: number; decided: number };
export type QueuePayload = {
  name: string;
  annotator: string;
  items: Item[];
  decisions: Record<string, Values>;
};

async function json<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error ?? `${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export const listQueues = () =>
  fetch("/api/review/queues").then(json<QueueSummary[]>);

export const loadQueue = (name: string) =>
  fetch(`/api/review/queues/${encodeURIComponent(name)}`).then(
    json<QueuePayload>,
  );

export const saveDecision = (
  name: string,
  itemId: string,
  values: Values,
  elapsedMs: number,
) =>
  fetch(`/api/review/queues/${encodeURIComponent(name)}/decisions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      item_id: itemId,
      values,
      elapsed_ms: Math.round(elapsedMs),
    }),
  }).then(json<{ ok: boolean }>);
