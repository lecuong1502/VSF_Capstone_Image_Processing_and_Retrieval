// Keyboard state machine for /review – pure, so it is unit-tested without a browser.
//
// Option key  set the active field, then move to the next field; after the last field the item
//             is submitted once every field has a value (one key per single-field item)
// Enter       submit when every field has a value (confirms pre-filled defaults)
// ↑ / ↓       change the active field
// Backspace/← previous item (shows its saved decision, re-submitting revises it)
// →           skip to the next item without deciding

export type Option = { value: string; key: string; label: string };
export type Field = {
  name: string;
  label: string;
  options: Option[];
  default: string | null;
};
export type Item = {
  item_id: string;
  mode: "moderation" | "kg" | "qrels";
  title: string;
  image_url: string | null;
  reference_urls: string[];
  context: Record<string, string>;
  fields: Field[];
};
export type Values = Record<string, string>;
export type Submission = { item_id: string; values: Values };

export type Session = {
  items: Item[];
  index: number; // === items.length when the queue is done
  active: number; // active field of the current item
  values: Values; // answers for the current item so far
  decisions: Record<string, Values>;
};

function load(session: Session, index: number): Session {
  const item = session.items[index];
  const values: Values = {
    ...(item ? initialValues(item, session.decisions) : {}),
  };
  return { ...session, index, active: 0, values };
}

function initialValues(item: Item, decisions: Record<string, Values>): Values {
  const saved = decisions[item.item_id];
  if (saved) return saved;
  const values: Values = {};
  for (const field of item.fields)
    if (field.default !== null) values[field.name] = field.default;
  return values;
}

export function start(
  items: Item[],
  decisions: Record<string, Values>,
): Session {
  const first = items.findIndex((item) => !(item.item_id in decisions));
  const session = { items, index: 0, active: 0, values: {}, decisions };
  return load(session, first === -1 ? items.length : first);
}

const complete = (item: Item, values: Values) =>
  item.fields.every((f) => f.name in values);

function submit(
  session: Session,
  item: Item,
): { session: Session; submit: Submission } {
  const values = { ...session.values };
  const decisions = { ...session.decisions, [item.item_id]: values };
  return {
    session: load({ ...session, decisions }, session.index + 1),
    submit: { item_id: item.item_id, values },
  };
}

export function onKey(
  session: Session,
  rawKey: string,
): { session: Session; submit?: Submission } {
  const item = session.items[session.index];
  if (rawKey === "Backspace" || rawKey === "ArrowLeft") {
    return {
      session: session.index > 0 ? load(session, session.index - 1) : session,
    };
  }
  if (!item) return { session };
  const last = item.fields.length - 1;
  switch (rawKey) {
    case "ArrowRight":
      return { session: load(session, session.index + 1) };
    case "ArrowDown":
      return {
        session: { ...session, active: Math.min(session.active + 1, last) },
      };
    case "ArrowUp":
      return {
        session: { ...session, active: Math.max(session.active - 1, 0) },
      };
    case "Enter":
      return complete(item, session.values)
        ? submit(session, item)
        : { session };
  }
  const key = rawKey.length === 1 ? rawKey.toLowerCase() : rawKey;
  const field = item.fields[session.active];
  const option = field.options.find((o) => o.key === key);
  if (!option) return { session };
  const next = {
    ...session,
    values: { ...session.values, [field.name]: option.value },
  };
  if (session.active < last)
    return { session: { ...next, active: session.active + 1 } };
  return complete(item, next.values) ? submit(next, item) : { session: next };
}
