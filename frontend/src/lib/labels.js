// Display names + helpers for the model's categorical values.

export const CONCEPTS = {
  deep_pass: { label: "Deep Pass", type: "pass" },
  short_pass: { label: "Short Pass", type: "pass" },
  screen_pass: { label: "Screen Pass", type: "pass" },
  pa_deep_pass: { label: "PA Deep Pass", type: "pass" },
  pa_short_pass: { label: "PA Short Pass", type: "pass" },
  inside_run: { label: "Inside Run", type: "run" },
  off_tackle_run: { label: "Off-Tackle Run", type: "run" },
  outside_run: { label: "Outside Run", type: "run" },
  qb_sneak: { label: "QB Sneak", type: "run" },
  trick_play: { label: "Trick Play", type: "trick" },
};

export function conceptLabel(concept) {
  return CONCEPTS[concept]?.label ?? concept;
}

export function conceptType(concept) {
  return CONCEPTS[concept]?.type ?? "pass";
}

export function formationLabel(formation) {
  if (!formation) return "Auto";
  return formation
    .replace("under_center", "Under Center")
    .replace("shotgun", "Shotgun")
    .replace("pistol", "Pistol")
    .replace("empty", "Empty")
    .replace("_1back", " · 1 Back")
    .replace("_2back", " · 2 Back");
}

export function personnelLabel(code) {
  if (!code) return "Auto";
  const rb = Number(code[0]);
  const te = Number(code[1]);
  const wr = Math.max(0, 5 - rb - te);
  return `${code} — ${rb} RB · ${te} TE · ${wr} WR`;
}

/** "BUF 35" / "Midfield" / "NYJ 38" from yards-to-opponent-end-zone. */
export function fieldPositionLabel(yardline100, posteam, defteam) {
  if (yardline100 === 50) return "Midfield";
  if (yardline100 < 50) return `${defteam} ${yardline100}`;
  return `${posteam} ${100 - yardline100}`;
}

export function pct(x) {
  return `${(x * 100).toFixed(0)}%`;
}

export function epa(x) {
  if (x === null || x === undefined) return "—";
  return `${x >= 0 ? "+" : ""}${x.toFixed(2)}`;
}
