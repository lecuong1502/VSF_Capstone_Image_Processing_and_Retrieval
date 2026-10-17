import { describe, expect, it } from "vitest";
import { onKey, start, type Field, type Item, type Session } from "./session";

const yesNo = (
  name: string,
  def: string | null = null,
  toggle: string | null = null,
): Field => ({
  name,
  label: name,
  default: def,
  toggle_key: toggle,
  options: [
    { value: "yes", key: "y", label: "yes" },
    { value: "no", key: "n", label: "no" },
  ],
});

const item = (id: string, fields: Field[]): Item => ({
  item_id: id,
  mode: "moderation",
  title: id,
  image_url: null,
  reference_urls: [],
  context: {},
  scores: {},
  model_version: null,
  fields,
});

const press = (session: Session, ...keys: string[]) => {
  const submitted: { item_id: string; values: Record<string, string> }[] = [];
  for (const key of keys) {
    const result = onKey(session, key);
    session = result.session;
    if (result.submit) submitted.push(result.submit);
  }
  return { session, submitted };
};

describe("review session", () => {
  it("starts at the first undecided item with saved or default values", () => {
    const items = [item("a", [yesNo("r")]), item("b", [yesNo("r", "no")])];
    const s = start(items, { a: { r: "yes" } });
    expect(s.index).toBe(1);
    expect(s.values).toEqual({ r: "no" });
  });

  it("decides a single-field item with one key and moves on", () => {
    const items = [item("a", [yesNo("r")]), item("b", [yesNo("r")])];
    const { session, submitted } = press(start(items, {}), "n", "y");
    expect(submitted).toEqual([
      { item_id: "a", values: { r: "no" } },
      { item_id: "b", values: { r: "yes" } },
    ]);
    expect(session.index).toBe(2); // done
    expect(session.decisions).toEqual({ a: { r: "no" }, b: { r: "yes" } });
  });

  it("fills blind multi-field items field by field and submits after the last", () => {
    const items = [item("a", [yesNo("w"), yesNo("t"), yesNo("p")])];
    const partial = press(start(items, {}), "y", "n");
    expect(partial.submitted).toEqual([]);
    expect(partial.session.active).toBe(2);
    expect(press(partial.session, "n").submitted).toEqual([
      { item_id: "a", values: { w: "yes", t: "no", p: "no" } },
    ]);
  });

  it("confirms defaults with Enter and edits one field with arrows", () => {
    const items = [item("a", [yesNo("w", "no"), yesNo("t", "no")])];
    const { submitted } = press(start(items, {}), "ArrowDown", "y", "Enter");
    expect(submitted).toEqual([
      { item_id: "a", values: { w: "no", t: "yes" } },
    ]);
  });

  it("Enter does nothing until every field has a value", () => {
    const items = [item("a", [yesNo("w"), yesNo("t")])];
    const { session, submitted } = press(start(items, {}), "y", "Enter");
    expect(submitted).toEqual([]);
    expect(session.index).toBe(0);
  });

  it("goes back to revise and skips forward without deciding", () => {
    const items = [
      item("a", [yesNo("r")]),
      item("b", [yesNo("r")]),
      item("c", [yesNo("r")]),
    ];
    let { session } = press(start(items, {}), "y");
    ({ session } = press(session, "Backspace"));
    expect(session.index).toBe(0);
    expect(session.values).toEqual({ r: "yes" }); // the saved decision
    const revised = press(session, "n");
    expect(revised.submitted).toEqual([{ item_id: "a", values: { r: "no" } }]);
    const skipped = press(revised.session, "ArrowRight");
    expect(skipped.session.index).toBe(2);
    expect(skipped.submitted).toEqual([]);
  });

  it("flags default to no, flip with their toggle key, and the last answer submits", () => {
    const items = [
      item("a", [yesNo("w", "no", "w"), yesNo("t", "no", "t"), yesNo("r")]),
      item("b", [yesNo("w", "no", "w"), yesNo("t", "no", "t"), yesNo("r")]),
    ];
    const s = start(items, {});
    expect(s.active).toBe(2); // the first field without a value
    const one = press(s, "y"); // a typical image: one key
    expect(one.submitted).toEqual([
      { item_id: "a", values: { w: "no", t: "no", r: "yes" } },
    ]);
    expect(one.session.active).toBe(2);
    const flagged = press(one.session, "t", "t", "w", "n");
    expect(flagged.submitted).toEqual([
      { item_id: "b", values: { w: "yes", t: "no", r: "no" } },
    ]);
  });

  it("ignores unknown keys and upper-case letters map to their key", () => {
    const items = [item("a", [yesNo("r")])];
    expect(press(start(items, {}), "x").submitted).toEqual([]);
    expect(press(start(items, {}), "N").submitted).toEqual([
      { item_id: "a", values: { r: "no" } },
    ]);
  });
});
