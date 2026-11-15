"""Minimal stdlib .xlsx -> .csv converter (no third-party packages). Usage: python -I xlsx_to_csv.py <in.xlsx> <out_dir>"""
import sys, zipfile, csv, re, os, xml.etree.ElementTree as ET
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
src, out = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(src)
ss = []
if "xl/sharedStrings.xml" in z.namelist():
    for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS):
        ss.append("".join(t.text or "" for t in si.iter("{%s}t" % NS["m"])))
wb = ET.fromstring(z.read("xl/workbook.xml"))
names = [s.get("name") for s in wb.find("m:sheets", NS)]
sheets = sorted(n for n in z.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml", n))
def col(ref):
    c = re.match(r"[A-Z]+", ref).group(); n = 0
    for ch in c: n = n * 26 + ord(ch) - 64
    return n - 1
for name, path in zip(names, sheets):
    rows = []
    for r in ET.fromstring(z.read(path)).iter("{%s}row" % NS["m"]):
        vals = {}
        for c in r.findall("m:c", NS):
            v = c.find("m:v", NS); t = c.get("t")
            if v is None:
                isel = c.find("m:is", NS); val = "".join(x.text or "" for x in isel.iter("{%s}t" % NS["m"])) if isel is not None else ""
            else:
                val = ss[int(v.text)] if t == "s" else v.text
            vals[col(c.get("r"))] = val
        if vals: rows.append([vals.get(i, "") for i in range(max(vals) + 1)])
    fn = os.path.join(out, os.path.basename(src).replace(".xlsx", "") + "__" + re.sub(r"\W+", "_", name) + ".csv")
    with open(fn, "w", newline="") as f: csv.writer(f).writerows(rows)
    print(fn, len(rows), "rows")
