"""Reading the CVAT COCO export.

Three things about this particular export that a generic COCO reader gets wrong:

1. Every annotation carries `iscrowd: 1`. CVAT stamps that on RLE segmentations,
   and most COCO tooling *silently drops* crowd annotations -- you get an empty
   dataset and no error. We normalise it to 0 on load.
2. The segmentations are uncompressed RLE (`counts` is a list of ints, runs in
   column-major order starting on background), not the compressed byte string
   `pycocotools` expects. The decoder here is 15 lines and removes the
   dependency entirely; `verify_dataset.py` checks it against the exporter's own
   `area` and `bbox` fields.
3. 7.5% of masks carry a detached speck a few dozen pixels across, left over
   from the interactive segmentation tool. Keeping the largest connected
   component removes every one; the count of what was dropped is recorded per
   mask rather than thrown away.

Masks are decoded lazily, one capture at a time -- decoding all 4381 at once
would be ~2.8 GB of booleans.
"""
import json
import re
from dataclasses import dataclass

import cv2
import numpy as np

FILE_NAME_RE = re.compile(
    r"preview_coco/(?P<mode>\w+)_images/day(?P<day>\d+)/dish(?P<dish>\d+)/(?P<side>\w+)/")


@dataclass(frozen=True)
class CaptureKey:
    mode: str
    day: int
    dish: int
    side: str

    def __str__(self):
        return f"{self.mode[:5]} day{self.day} dish{self.dish} {self.side}"


@dataclass
class Annotation:
    ann_id: int
    image_id: int
    counts: list
    size: tuple          # (h, w) == (640, n_lines), i.e. cube layout
    area: float          # the exporter's own value, used as a decoder check
    bbox: list


@dataclass
class CaptureAnns:
    key: CaptureKey
    image_id: int
    height: int          # 640, the cube's spatial axis
    width: int           # n_lines
    file_name: str
    anns: list


def decode_rle(counts, size):
    """COCO uncompressed RLE -> bool (h, w). Runs are column-major, start at 0."""
    h, w = size
    flat = np.zeros(h * w, np.uint8)
    pos, val = 0, 0
    for n in counts:
        if val:
            flat[pos:pos + n] = 1
        pos += n
        val ^= 1
    return flat.reshape((h, w), order="F").astype(bool)


def keep_largest_component(mask):
    """-> (cleaned mask, n_components_before, dropped_px).

    The interactive tool occasionally leaves a detached speck inside the mask's
    bounding box. It is never the kernel, and it drags the mask-mean spectrum
    towards whatever it sits on.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 2:                                   # background + one component
        return mask, max(n - 1, 0), 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = int(areas.argmax()) + 1
    cleaned = labels == keep
    return cleaned, n - 1, int(mask.sum() - cleaned.sum())


def erode(mask, px):
    """Shrink a mask by `px` pixels from its boundary.

    Some masks clip a sliver of the plastic retaining clip at the kernel's
    edge. A few clip pixels are a large fraction of a 1200-pixel kernel and
    they are plastic, not barley -- exactly the kind of contamination a variety
    model would happily learn. Eroding costs the outer ring of genuine kernel
    too, which is the cheaper mistake: the interior is what carries the
    spectrum, and the boundary ring is where the annotator was least certain
    anyway.

    Applied in cube layout, before the warp, so `px` means annotation pixels.
    Erosion can pinch a mask into two lobes, so the caller takes the largest
    component again afterwards.
    """
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.erode(mask.astype(np.uint8), k, iterations=1).astype(bool)


def load(coco_json, hours_map=None):
    """-> {CaptureKey: CaptureAnns}, restricted to the days in `hours_map`.

    Captures with zero annotations are dropped and reported by the caller; in
    the current export those are exactly the day-2 captures, which are out of
    scope anyway.
    """
    doc = json.loads(coco_json.read_text())

    cats = {c["id"]: c["name"] for c in doc["categories"]}
    if set(cats.values()) != {"KERNEL"}:
        raise SystemExit(f"unexpected categories in {coco_json}: {cats}")

    by_image = {}
    for a in doc["annotations"]:
        seg = a["segmentation"]
        if not isinstance(seg, dict) or not isinstance(seg.get("counts"), list):
            raise SystemExit(
                f"annotation {a['id']} is not uncompressed RLE -- this reader "
                "handles CVAT's RLE export only; re-export as COCO 1.0")
        by_image.setdefault(a["image_id"], []).append(Annotation(
            ann_id=a["id"], image_id=a["image_id"], counts=seg["counts"],
            size=tuple(seg["size"]), area=float(a["area"]), bbox=list(a["bbox"])))

    out = {}
    for im in doc["images"]:
        m = FILE_NAME_RE.match(im["file_name"])
        if not m:
            raise SystemExit(f"cannot parse capture identity from {im['file_name']!r}")
        day = int(m["day"])
        if hours_map is not None and day not in hours_map:
            continue
        key = CaptureKey(m["mode"], day, int(m["dish"]), m["side"])
        anns = by_image.get(im["id"], [])
        if not anns:
            continue
        if key in out:
            raise SystemExit(f"two COCO images map to the same capture: {key}")
        out[key] = CaptureAnns(key=key, image_id=im["id"], height=im["height"],
                               width=im["width"], file_name=im["file_name"], anns=anns)
    return out
