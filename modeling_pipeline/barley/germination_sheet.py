"""Read the germination annotator's spreadsheet.

    germination_label_annotator.xlsx
      row 1     header: DISH/GRAIN, then kernel indices 0..21
      column A  dish number, 0..24
      cell      first germination day, or -1 for never, or BLANK for not scored

The kernel index is gridfit's `cell_index`, the same 0..21 numbering stamped on
`labels/germination_photos/index_reference_{dorsal,ventral}.png`. It is
positional over the fixed plate model, so index 3 names the same physical well
on every dish whether or not that well holds a kernel.

BLANK IS NOT -1
-1 means "scored, and it never germinated" -- a right-censored observation that
belongs in training. Blank means "nobody has looked at this dish yet", which is
a missing label and must be dropped. Conflating them would train the model to
predict dormancy from unscored plates.

WHY NOT openpyxl
Neither openpyxl nor pandas is installed, and verify_dataset.py deliberately
runs on the system python with only numpy/scipy/cv2. An .xlsx is a zip of XML
and the part needed here -- a single rectangular sheet of numbers -- is about
forty lines. Adding a dependency to the gate is the more expensive option.
"""
import re
import zipfile
from xml.etree import ElementTree as ET

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
NEVER = -1


def _col_letters(ref):
    return re.match(r"[A-Z]+", ref).group()


def _col_index(letters):
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n - 1


def read_cells(path):
    """-> ({(row, col_index): text}, sheet_name). Raw, unvalidated."""
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            shared = [(t.text or "") for t in
                      ET.fromstring(z.read("xl/sharedStrings.xml")).iter(NS + "t")]
        names = [s.get("name") for s in
                 ET.fromstring(z.read("xl/workbook.xml")).iter(NS + "sheet")]
        root = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))

    cells = {}
    for r in root.iter(NS + "row"):
        for c in r.iter(NS + "c"):
            ref = c.get("r")
            v = c.find(NS + "v")
            if v is None or v.text is None:
                continue
            text = shared[int(v.text)] if c.get("t") == "s" else v.text
            text = (text or "").strip()
            if text == "":
                continue
            row = int(re.search(r"\d+", ref).group())
            cells[(row, _col_index(_col_letters(ref)))] = text
    return cells, (names[0] if names else None)


def read_grid(path, n_kernels=22, n_dishes=25):
    """-> {(dish, cell_index): day or NEVER}, for SCORED cells only.

    Unscored cells are simply absent from the mapping -- the caller decides what
    a missing kernel means, and it is never the same as NEVER.
    """
    cells, sheet = read_cells(path)
    if not cells:
        raise SystemExit(f"{path}: first worksheet is empty")

    # Header row names the kernel indices. Validate it rather than assuming
    # column B is index 0: a spreadsheet that gains a column silently shifts
    # every label by one well, which nothing downstream could detect.
    header = {}
    for col in range(1, n_kernels + 8):
        txt = cells.get((1, col))
        if txt is None:
            continue
        try:
            header[col] = int(txt)
        except ValueError:
            continue
    if not header:
        raise SystemExit(
            f"{path}: header row 1 has no integer kernel indices. Expected "
            f"'DISH/GRAIN' in A1 then 0..{n_kernels - 1}.")
    want = set(range(n_kernels))
    got = set(header.values())
    if got != want:
        raise SystemExit(
            f"{path}: header row is {sorted(got)}, expected 0..{n_kernels - 1}. "
            f"Missing {sorted(want - got)}, unexpected {sorted(got - want)}.")

    out = {}
    seen_dishes = set()
    for row in range(2, 2 + n_dishes + 5):
        raw = cells.get((row, 0))
        if raw is None:
            continue
        try:
            dish = int(raw)
        except ValueError:
            raise SystemExit(f"{path}: row {row} column A is {raw!r}, not a dish "
                             f"number")
        if dish in seen_dishes:
            raise SystemExit(f"{path}: dish {dish} appears on more than one row")
        seen_dishes.add(dish)
        if not 0 <= dish < n_dishes:
            raise SystemExit(f"{path}: row {row} has dish {dish}; expected "
                             f"0..{n_dishes - 1}")
        for col, kidx in header.items():
            txt = cells.get((row, col))
            if txt is None:
                continue                     # not scored; absent, not NEVER
            try:
                v = int(float(txt))
            except ValueError:
                raise SystemExit(
                    f"{path}: dish {dish} kernel {kidx} is {txt!r}, which is "
                    f"neither a day number nor {NEVER}")
            out[(dish, kidx)] = v
    return out, sheet
