"""Draw the fitted grid on the render, including when the fit failed.

A failed fit gets an overlay too -- with whatever was detected and the reasons
printed on it -- because that image is how a failure gets diagnosed. It is
labelled FAIL in red and carries no cell outlines, so it cannot be mistaken for
a usable grid.

Cells are outlined at exactly the footprint `cells.cell_mask` rasterises, so the
picture never promises a region the data does not contain.
"""
import cv2
import numpy as np

STATUS_COLOR = {"pass": (90, 220, 90), "needs_review": (0, 200, 255),
                "fail": (60, 60, 255)}
KIND_COLOR = {"kernel": (110, 230, 110), "rim": (130, 130, 130),
              "marker": (0, 190, 255)}
SEED_COLOR = (255, 140, 0)
FIDUCIAL_COLOR = (0, 0, 255)
PREDICTED_COLOR = (230, 80, 230)


def _text(img, s, org, scale=0.42, color=(255, 255, 0)):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw(gray, res, cells, seeds, title):
    """-> BGR image, same size as the render, so it stays pixel-aligned with it."""
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    status = res["status"]

    for s in seeds:
        x, y = s["center"]
        cv2.drawMarker(vis, (int(x), int(y)), SEED_COLOR, cv2.MARKER_CROSS, 9, 1)

    for cell in cells or []:
        pts = np.rint(np.asarray(cell["corners"], float)).astype(np.int32)
        cv2.polylines(vis, [pts], True, KIND_COLOR[cell["kind"]], 1, cv2.LINE_AA)
        cx, cy = cell["center"]
        if cell["kind"] == "kernel":
            _text(vis, cell["name"], (int(cx) - 22, int(cy) - 2), 0.42)
            _text(vis, f"#{cell['index']}", (int(cx) - 14, int(cy) + 14), 0.34,
                  (200, 255, 200))
        elif cell["kind"] == "marker":
            seen = cell["marker_id"]
            _text(vis, cell["name"], (int(cx) - 22, int(cy) - 2), 0.42, (0, 210, 255))
            _text(vis, f"id{seen}" if seen is not None else "id?",
                  (int(cx) - 12, int(cy) + 14), 0.36, (0, 210, 255))

    # Solid red: a fiducial the detector segmented on its own, whose position is
    # an independent check on the lattice. Dashed magenta: one read at the square
    # the lattice predicts, which confirms the code but says nothing about the
    # geometry -- drawn differently so the two are never read as the same claim.
    for mid, m in res.get("markers", {}).items():
        cv2.polylines(vis, [np.rint(np.asarray(m["corners"])).astype(np.int32)],
                      True, FIDUCIAL_COLOR, 2, cv2.LINE_AA)
        x, y = m["center"]
        _text(vis, f"id{mid}{'*' if m['exact'] else ''} {m['score']:.2f}",
              (int(x) - 26, int(y) - 44), 0.36, (120, 180, 255))
    for m in (res.get("predicted_markers") or {}).values():
        q = np.rint(np.asarray(m["corners"])).astype(np.int32)
        for i in range(4):
            a, b = q[i], q[(i + 1) % 4]
            for t in (0.0, 0.35, 0.70):
                p0 = (a + (b - a) * t).astype(int)
                p1 = (a + (b - a) * (t + 0.2)).astype(int)
                cv2.line(vis, tuple(p0), tuple(p1), PREDICTED_COLOR, 2, cv2.LINE_AA)
        x, y = m["center"]
        _text(vis, f"id{m['id']} pred {m['score']:.2f}/+{m['margin']:.2f}",
              (int(x) - 40, int(y) - 44), 0.34, PREDICTED_COLOR)

    header = [(title, (230, 230, 230))]
    header += [(f"{status.upper()}  {_summary(res)}", STATUS_COLOR[status])]
    header += [(f"! {r}", STATUS_COLOR["fail"]) for r in res.get("reasons", [])]
    header += [(f"~ {f}", STATUS_COLOR["needs_review"]) for f in res.get("flags", [])]
    header += [(f". {n}", (170, 170, 170)) for n in res.get("info", [])]

    wrapped = [(part, col) for text, col in header for part in _wrap(text, vis.shape[1])]
    pad = np.zeros((18 * len(wrapped) + 10, vis.shape[1], 3), np.uint8)
    for i, (s, col) in enumerate(wrapped):
        _text(pad, s, (6, 16 + i * 18), 0.40, col)
    return np.vstack([pad, vis])


def _wrap(text, width_px, scale=0.40):
    """Break on spaces so a long reason wraps instead of running off the frame."""
    room = max(int((width_px - 14) / max(cv2.getTextSize(
        "x", cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0], 1)), 20)
    out, line = [], ""
    for word in text.split(" "):
        if line and len(line) + 1 + len(word) > room:
            out.append(line)
            line = "   " + word
        else:
            line = f"{line} {word}".strip() if line else word
    return out + [line] if line else out


def _summary(res):
    g = res.get("geometry")
    p = res.get("pose")
    bits = [f"seeds={res.get('n_seeds', 0)}",
            f"matched={res.get('n_matched', 0)}/22",
            f"ids={res.get('marker_ids') or '-'}"]
    if g:
        bits += [f"px={g['pitch_x_px']:.1f}", f"py={g['pitch_y_px']:.1f}",
                 f"ang={g['angle_deg']:.1f}", g["chirality"][0].upper()]
    if res.get("rms_px") is not None:
        bits.append(f"rms={res['rms_px']:.2f}")
    if p:
        bits += [p["pose"], f"margin={p['margin']:.0f}"]
    off = res.get("marker_offsets_px") or {}
    bits += [f"m{k}={v:.0f}px" for k, v in sorted(off.items())]
    return "  ".join(bits)
