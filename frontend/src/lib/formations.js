// Turns the model's inputs (formation, personnel, defenders_in_box) into
// player coordinates for the field diagram.
//
// Coordinates are in *yards relative to the line of scrimmage*:
//   dx  - along the field; negative = offense side, positive = defense side
//   dy  - across the field; 0 = middle, negative = left, positive = right
// The Field component shifts everything to the right spot on the field.

export const FIELD_WIDTH = 53.3; // sideline to sideline, in yards

function parsePersonnel(code) {
  const rb = Number(code?.[0]);
  const te = Number(code?.[1]);
  if (!Number.isInteger(rb) || !Number.isInteger(te)) return { rb: 1, te: 1 };
  return { rb, te };
}

/**
 * Eleven offensive players for a formation + personnel grouping.
 * Returns [{ id, label, role, dx, dy }] - role is one of
 * ol | qb | rb | te | wr (used for colors and route art).
 */
export function offensePlayers(formation = "shotgun_1back", personnel = "11") {
  const { rb, te } = parsePersonnel(personnel);
  const wr = Math.max(0, 5 - rb - te);
  const align =
    formation === "empty" ? "shotgun" : formation.replace(/_(1|2)back$/, "");
  const backs = formation === "empty" ? 0 : formation.endsWith("2back") ? 2 : 1;

  const players = [];
  const add = (label, role, dx, dy) =>
    players.push({ id: `${role}${players.length}`, label, role, dx, dy });

  // Offensive line - always the same five.
  add("C", "ol", -0.8, 0);
  add("LG", "ol", -0.8, -1.7);
  add("RG", "ol", -0.8, 1.7);
  add("LT", "ol", -0.8, -3.4);
  add("RT", "ol", -0.8, 3.4);

  // Who fills the backfield: RBs first, then TEs (e.g. "11" with a
  // 2-back formation puts the TE at fullback). Leftover RBs flex out
  // as receivers, leftover TEs attach to the line.
  const rbInBackfield = Math.min(rb, backs);
  const teInBackfield = Math.min(te, backs - rbInBackfield);
  const slotRB = rb - rbInBackfield;
  const attachedTE = te - teInBackfield;

  // Tight ends on the line: right, left, then wings.
  const tePos = [
    [-0.8, 5.1],
    [-0.8, -5.1],
    [-1.8, 6.6],
    [-1.8, -6.6],
  ];
  for (let i = 0; i < attachedTE && i < tePos.length; i++)
    add("TE", "te", tePos[i][0], tePos[i][1]);

  // Receivers: wide first, then slots. Flexed RBs take the last spots.
  const wrPos = [
    [-1.2, 21],
    [-1.2, -21],
    [-2.2, 14.5],
    [-2.2, -14.5],
    [-2.2, 9.5],
  ];
  const receivers = wr + slotRB;
  for (let i = 0; i < receivers && i < wrPos.length; i++) {
    const isRB = i >= wr;
    add(isRB ? "RB" : "WR", isRB ? "rb" : "wr", wrPos[i][0], wrPos[i][1]);
  }

  // Quarterback depth comes from the formation family.
  const qbDepth = { under_center: -2, pistol: -4.5, shotgun: -6 }[align] ?? -6;
  add("QB", "qb", qbDepth, 0);

  // Backfield spots per alignment (first spot fills first).
  let backfieldPos;
  if (align === "shotgun")
    backfieldPos = backs === 2 ? [[-6, 3.4], [-6, -3.4]] : [[-6, 3.4]];
  else if (align === "pistol")
    backfieldPos = backs === 2 ? [[-7.2, 0], [-5, 3]] : [[-7.2, 0]];
  // under center: I-formation when there are two backs
  else backfieldPos = backs === 2 ? [[-5, 0], [-7.8, 0]] : [[-6.8, 0]];

  for (let i = 0; i < rbInBackfield + teInBackfield && i < backfieldPos.length; i++) {
    const isTE = i >= rbInBackfield;
    const isFB = !isTE && backs === 2 && align === "under_center" && i === 0;
    add(isTE ? "TE" : isFB ? "FB" : "RB", isTE ? "te" : "rb",
      backfieldPos[i][0], backfieldPos[i][1]);
  }

  return players;
}

/**
 * Eleven defenders for a box count, reacting to where the offense lined
 * up (corners walk out over the wide receivers).
 */
export function defensePlayers(box = 7, offense = []) {
  const players = [];
  const add = (label, role, dx, dy) =>
    players.push({ id: `${role}${players.length}`, label, role, dx, dy });

  // Down linemen (max 4-man front), then linebackers fill the box.
  const dlCount = Math.min(4, box);
  const dlSpots = {
    0: [],
    1: [0],
    2: [-1.8, 1.8],
    3: [-3, 0, 3],
    4: [-4, -1.4, 1.4, 4],
  }[dlCount];
  dlSpots.forEach((dy) => add("DL", "dl", 1.2, dy));

  const lbCount = Math.max(0, Math.min(7, box - dlCount));
  const lbSpots = {
    0: [],
    1: [0],
    2: [-3.4, 3.4],
    3: [-4.5, 0, 4.5],
    4: [-7, -2.4, 2.4, 7],
    5: [-8, -4, 0, 4, 8],
    6: [-9, -5.4, -1.8, 1.8, 5.4, 9],
    7: [-9, -6, -3, 0, 3, 6, 9],
  }[lbCount];
  lbSpots.forEach((dy) => add("LB", "lb", 4.3, dy));

  // The rest play in the secondary: corners over the widest receivers,
  // then two deep safeties, then nickel/dime over the slots.
  const secondary = Math.max(0, 11 - box);
  const wideouts = offense
    .filter((p) => p.role === "wr" || (p.role === "rb" && Math.abs(p.dy) > 8))
    .sort((a, b) => Math.abs(b.dy) - Math.abs(a.dy));
  const over = (i, fallbackDy) => wideouts[i]?.dy ?? fallbackDy;

  const spots = [
    ["CB", "db", 1.8, over(0, 19)],
    ["CB", "db", 1.8, over(1, -19)],
    ["FS", "db", 12, secondary >= 4 ? 6.5 : 0],
    ["SS", "db", 12, -6.5],
    ["NB", "db", 3, over(2, 10)],
    ["DB", "db", 3, over(3, -10)],
    ["DB", "db", 9, 0],
    ["DB", "db", 14, 0],
  ];
  for (let i = 0; i < secondary && i < spots.length; i++)
    add(spots[i][0], spots[i][1], spots[i][2], spots[i][3]);

  return players;
}

/**
 * Route/run art for the recommended concept: arrows drawn from the
 * relevant players. Returns [{ from: player, points: [[dx,dy], ...],
 * dashed }] - points are offsets *from that player's position*.
 */
export function playArt(concept, offense) {
  if (!concept) return [];
  const arrows = [];
  const wrs = offense
    .filter((p) => p.role === "wr")
    .sort((a, b) => Math.abs(b.dy) - Math.abs(a.dy));
  const slots = offense
    .filter((p) => (p.role === "te" || p.role === "rb" || p.role === "wr") &&
      p.dx <= -1.5 && Math.abs(p.dy) > 4)
    .sort((a, b) => Math.abs(a.dy) - Math.abs(b.dy));
  const back =
    offense.find((p) => p.role === "rb" && Math.abs(p.dy) <= 8 && p.dx < -3) ??
    offense.find((p) => p.role === "te" && p.dx < -3);
  const qb = offense.find((p) => p.role === "qb");

  const playAction = concept.startsWith("pa_");
  if (playAction && back && qb) {
    // the run fake
    arrows.push({ from: qb, points: [[0, 0], [back.dx - qb.dx + 1, back.dy - qb.dy]], dashed: true });
  }

  const side = (p) => (p.dy >= 0 ? 1 : -1);

  switch (concept.replace(/^pa_/, "")) {
    case "deep_pass":
      wrs.slice(0, 2).forEach((p) =>
        arrows.push({ from: p, points: [[0, 0], [12, -side(p) * 1], [20, -side(p) * 3]], dashed: playAction })
      );
      break;
    case "short_pass":
      (slots.length ? slots.slice(0, 2) : wrs.slice(0, 2)).forEach((p) =>
        arrows.push({ from: p, points: [[0, 0], [5, 0], [6.5, -side(p) * 4]], dashed: playAction })
      );
      break;
    case "screen_pass": {
      const target = back ?? wrs[0];
      if (target)
        arrows.push({
          from: target,
          points: [[0, 0], [-1.5, side(target) * 5], [2, side(target) * 7]],
          dashed: true,
        });
      break;
    }
    case "inside_run":
      if (back) arrows.push({ from: back, points: [[0, 0], [3, -back.dy], [8, -back.dy]] });
      break;
    case "off_tackle_run":
      if (back) arrows.push({ from: back, points: [[0, 0], [2.5, 4 - back.dy], [8, 5.5 - back.dy]] });
      break;
    case "outside_run":
      if (back) arrows.push({ from: back, points: [[0, 0], [1, 10 - back.dy], [7, 15 - back.dy]] });
      break;
    case "qb_sneak":
      if (qb) arrows.push({ from: qb, points: [[0, 0], [qb.dx * -1 + 2.5, 0]] });
      break;
    case "trick_play":
      if (wrs[0] && qb)
        arrows.push({
          from: wrs[0],
          points: [[0, 0], [qb.dx - wrs[0].dx, -wrs[0].dy * 0.9], [12, -wrs[0].dy * 0.9 + side(wrs[0]) * 2]],
          dashed: true,
        });
      break;
    default:
      break;
  }
  return arrows;
}
