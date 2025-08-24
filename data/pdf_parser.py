# data/extract.py
import argparse, os, pathlib, re, sys
from typing import Dict, List, Optional, Tuple

IN_DIR  = pathlib.Path("data/financial_reports_actual")
OUT_DIR = pathlib.Path("data/financial_reports_clean")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- OCR deps (required) ---
from PIL import Image, ImageOps, ImageFilter
try:
    from pdf2image import convert_from_path
    import pytesseract
except Exception as e:
    print("ERROR: OCR dependencies missing. Install poppler + tesseract, then:\n"
          "  pip install pdf2image pytesseract pillow\n"
          f"Details: {e}", file=sys.stderr)
    sys.exit(1)

# Optional: set these if Windows or custom locations
POPPLER_PATH = os.environ.get("POPPLER_PATH", None)  # e.g., r"C:\poppler-24.07.0\Library\bin"
TESSERACT_CMD = os.environ.get("TESSERACT_CMD", None)  # e.g., r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# ---------- utils ----------
AMT_PAT = r"(?:£\s*)?\(?-?\d{1,3}(?:,\d{3})*(?:\.\d+)?\)?"

def norm_ws(t: str) -> str:
    t = (t or "").replace("\r", "\n")
    t = t.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    t = re.sub(r"[ \t]+", " ", t)
    return t

def currency_norm(raw: Optional[str], force_positive: bool=False) -> Optional[str]:
    if not raw: return None
    s = raw.strip()
    neg = "(" in s and ")" in s
    s = re.sub(r"[^\d,.\-]", "", s)
    if s.startswith("-"):
        neg = True
        s = s[1:]
    if not s: return None
    out = f"£{s}"
    if force_positive:
        return out
    return f"-{out}" if neg else out

def first_amount(line: str, force_positive: bool=False) -> Optional[str]:
    m = re.search(AMT_PAT, line)
    return currency_norm(m.group(0), force_positive=force_positive) if m else None

def last_amount(line: str, force_positive: bool=False) -> Optional[str]:
    m = list(re.finditer(AMT_PAT, line))
    return currency_norm(m[-1].group(0), force_positive=force_positive) if m else None

def ocr_pdf_to_text(pdf_path: pathlib.Path, dpi: int = 300) -> str:
    # Convert every page to an image, lightly enhance, run Tesseract
    if POPPLER_PATH:
        imgs = convert_from_path(str(pdf_path), dpi=dpi, poppler_path=POPPLER_PATH)
    else:
        imgs = convert_from_path(str(pdf_path), dpi=dpi)

    chunks: List[str] = []
    for img in imgs:
        # light pre-processing: grayscale → contrast → de-noise → sharp
        g = ImageOps.grayscale(img)
        g = ImageOps.autocontrast(g)
        g = g.filter(ImageFilter.MedianFilter(size=3))
        g = g.filter(ImageFilter.SHARPEN)
        txt = pytesseract.image_to_string(
            g,
            lang="eng",
            config="--oem 3 --psm 6"  # assume single uniform block of text
        ) or ""
        chunks.append(norm_ws(txt))
    return "\n\n".join(chunks)

def section(full_text: str, start_pat: str, end_pats: List[str]) -> Optional[str]:
    s = re.search(start_pat, full_text, flags=re.I|re.S)
    if not s:
        return None
    end = len(full_text)
    for ep in end_pats:
        e = re.search(ep, full_text[s.end():], flags=re.I|re.S)
        if e:
            end = s.end() + e.start()
            break
    return full_text[s.start():end]

def extract_by_labels(sec_text: str,
                      labels: List[Tuple[str, str, bool]]) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {outlab: None for _, outlab, _ in labels}
    if not sec_text:
        return out
    lines = [ln.strip() for ln in sec_text.split("\n") if ln.strip()]
    for search_lab, outlab, force_pos in labels:
        pat = re.compile(r"\b" + re.escape(search_lab).replace(r"\ ", r"\s+") + r"\b", re.I)
        for ln in lines:
            if pat.search(ln):
                val = first_amount(ln, force_positive=force_pos)
                if val:
                    out[outlab] = val
                    break
    return out

# ---------- fixed logic tuned to your PDFs ----------
SOFA_LABELS = [
    ("Donated services",                    "Donated services",                False),
    ("Income from Investments",             "Investment income",               False),
    ("Refunds received",                    "Refunds received",                False),
    ("Total Income and Endowments",         "Total Income and Endowments",     False),
    # Expenditure shown as positive in your target .txt
    ("Management Fees",                     "Management Fees",                 True),
    ("Charitable Activities",               "Charitable Activities",           True),
    ("Audit Fees",                          "Audit Fees",                      True),
    ("Total Expenditure",                   "Total Expenditure",               True),
    # Net rows keep their sign
    ("Net Income (Expenditure)",            "Net Income (Expenditure)",        False),
    ("Net Income",                          "Net Income",                      False),
    ("Net (Losses) Gains on Investments",   "Net (Losses) Gains on Investments", False),
    ("Net Movement in Funds",               "Net Movement in Funds",           False),
    ("Total Funds carried forward",         "Total Funds carried forward",     False),
]

BAL_LABELS = [
    ("Investments",        "Investments",        False),
    ("Debtors",            "Debtors",            False),
    ("Creditors",          "Creditors",          True),   # show positive
    ("Total Net Assets",   "Total Net Assets",   False),
    ("Unrestricted funds", "Unrestricted Funds", False),
]

def parse_all(full_text: str) -> Dict[str, Dict]:
    text = norm_ws(full_text)

    sofa = section(text, r"STATEMENT OF FINANCIAL ACTIVITIES", [r"\nBALANCE SHEET", r"\nNOTES TO THE FINANCIAL STATEMENTS"])
    bal  = section(text, r"\nBALANCE SHEET", [r"\nNOTES TO THE FINANCIAL STATEMENTS", r"\n1\s+ACCOUNTING POLICIES"])
    notes = section(text, r"NOTES TO THE FINANCIAL STATEMENTS", [r"$"])

    S = extract_by_labels(sofa or "", SOFA_LABELS)
    # prefer whichever net label we found
    if not S.get("Net Income") and S.get("Net Income (Expenditure)"):
        pass
    elif not S.get("Net Income (Expenditure)") and S.get("Net Income"):
        pass

    B = extract_by_labels(bal or "", BAL_LABELS)

    # Note 3: Investment Income
    inv_inc = section(notes or "", r"\n3\s+INVESTMENT INCOME", [r"\n4\s", r"\n5\s", r"$"]) or ""
    NI = extract_by_labels(inv_inc, [
        ("Dividends - equities",      "Dividends - equities", False),
        ("Interest on cash deposits", "Interest on cash deposits", False),
        ("Total",                     "Total", False),
    ])

    # Note 4: Investment Management Costs
    mgmt = section(notes or "", r"\n4\s+INVESTMENT MANAGEMENT COSTS", [r"\n5\s", r"\n6\s", r"$"]) or ""
    MF = extract_by_labels(mgmt, [("Investment and ELC management fees", "Fees", False)])

    # Note 5: Costs of Charitable Activities → capture grants line(s)
    char = section(notes or "", r"\n5\s+COSTS OF CHARITABLE ACTIVITIES", [r"\n6\s", r"\n7\s", r"$"]) or ""
    grants_lines: List[str] = []
    total_grants = None
    if char:
        for ln in [l.strip() for l in char.split("\n") if l.strip()]:
            if re.search(r"\bGrants?\b", ln, flags=re.I):
                amt = first_amount(ln, force_positive=True)
                if amt:
                    ln = re.sub(AMT_PAT, amt, ln)
                grants_lines.append(ln)
            elif re.fullmatch(r"(?i)total.*", ln):
                total_grants = first_amount(ln, force_positive=True) or total_grants

    # Note 7: Investments – movement
    inv_mv = section(notes or "", r"\n7\s+INVESTMENTS", [r"\n8\s", r"\n9\s", r"$"]) or ""
    IM = {
        "Market Value at 1 April": None,
        "Profits from Sales": None,
        "Net loss on revaluations": None,
        "Net Gain overall": None,    # we'll relabel by sign later
        "Management Fees deducted": None,
        "Market Value at 31 March": None,
    }
    for ln in [l.strip() for l in inv_mv.split("\n") if l.strip()]:
        low = ln.lower()
        if "market value at 1 april" in low:
            IM["Market Value at 1 April"] = first_amount(ln, force_positive=True)
        elif "profits" in low and "sales" in low:
            IM["Profits from Sales"] = first_amount(ln, force_positive=True)
        elif ("fair value" in low) or ("revaluations" in low and "net" in low):
            IM["Net loss on revaluations"] = first_amount(ln, force_positive=False)
        elif "gains" in low and "investments" in low:
            IM["Net Gain overall"] = first_amount(ln, force_positive=False)
        elif "management fees deducted" in low:
            IM["Management Fees deducted"] = first_amount(ln, force_positive=False)
        elif "market value at 31 march" in low:
            IM["Market Value at 31 March"] = first_amount(ln, force_positive=True)

    # Note 8: Debtors
    debt = section(notes or "", r"\n8\s+DEBTORS", [r"\n9\s", r"\n10\s", r"$"]) or ""
    DB = {"Other Debtors": None}
    for ln in [l.strip() for l in debt.split("\n") if l.strip()]:
        if re.search(r"\bOther Debtors\b", ln, flags=re.I):
            DB["Other Debtors"] = first_amount(ln, force_positive=True)
            break

    # Note 9: Analysis of Charitable Funds → closing balance (last amount on General Funds line)
    funds = section(notes or "", r"\n9\s+ANALYSIS OF CHARITABLE FUNDS", [r"\n10\s", r"$"]) or ""
    CF = {"Closing Balance": None}
    for ln in [l.strip() for l in funds.split("\n") if l.strip()]:
        if re.search(r"\bGeneral Funds\b", ln, flags=re.I):
            CF["Closing Balance"] = last_amount(ln, force_positive=True)
            break

    return {
        "SoFA": S,
        "Balance": B,
        "Notes": {
            "Investment Income": NI,
            "Investment Management Costs": MF,
            "Costs of Charitable Activities": {"lines": grants_lines, "total": total_grants},
            "Investments – Movement": IM,
            "Debtors": DB,
            "Analysis of Charitable Funds": CF,
        },
    }

def infer_period_and_date(full_text: str, filename: str) -> Tuple[str, str]:
    m = re.search(r"YEAR ENDED 31 MARCH (\d{4})", full_text, flags=re.I)
    if m:
        y = int(m.group(1)); return f"{y-1}/{str(y)[-2:]}", f"31 March {y}"
    m2 = re.search(r"(20\d{2})", filename)
    if m2:
        y = int(m2.group(1)); return f"{y-1}/{str(y)[-2:]}", f"31 March {y}"
    return "Unknown", "Unknown"

def output_name(pdf_name: str) -> str:
    m = re.search(r"(20\d{2})", pdf_name)
    y = m.group(1) if m else "unknown"
    return f"drbruce_{y}.txt"

def format_txt(parsed: Dict[str, Dict], period: str, bal_date: str) -> str:
    S = parsed["SoFA"]; B = parsed["Balance"]; N = parsed["Notes"]
    lines: List[str] = []

    lines.append(f"===== STATEMENT OF FINANCIAL ACTIVITIES ({period}) =====")
    # income
    if S.get("Donated services"): lines.append(f"Donated services: {S['Donated services']}")
    if S.get("Investment income"): lines.append(f"Investment income: {S['Investment income']}")
    if S.get("Refunds received"):  lines.append(f"Refunds received: {S['Refunds received']}")
    if S.get("Total Income and Endowments"): lines.append(f"Total Income and Endowments: {S['Total Income and Endowments']}")
    lines.append("")
    # expenditure (positive)
    for k in ["Management Fees","Charitable Activities","Audit Fees","Total Expenditure"]:
        if S.get(k): lines.append(f"{k}: {S[k]}")
    lines.append("")
    # net
    if S.get("Net Income (Expenditure)"):
        lines.append(f"Net Income (Expenditure): {S['Net Income (Expenditure)']}")
    elif S.get("Net Income"):
        lines.append(f"Net Income: {S['Net Income']}")
    inv_net = S.get("Net (Losses) Gains on Investments")
    if inv_net:
        lab = "Net Loss on Investments" if inv_net.startswith("-") else "Net Gains on Investments"
        lines.append(f"{lab}: {inv_net}")
    if S.get("Net Movement in Funds"):       lines.append(f"Net Movement in Funds: {S['Net Movement in Funds']}")
    if S.get("Total Funds carried forward"): lines.append(f"Total Funds carried forward: {S['Total Funds carried forward']}")
    lines.append("")

    lines.append(f"===== BALANCE SHEET (As at {bal_date}) =====")
    for lab in ["Investments","Debtors","Creditors","Total Net Assets"]:
        v = B.get(lab if lab != "Unrestricted Funds" else "Unrestricted funds")
        if v: lines.append(f"{lab}: {v}")
    if B.get("Unrestricted Funds"): lines.append(f"Unrestricted Funds: {B['Unrestricted Funds']}")
    lines.append("")

    lines.append(f"===== NOTES TO THE FINANCIAL STATEMENTS ({period}) =====")
    ii = N["Investment Income"]
    if any(ii.values()):
        lines.append("Investment Income:")
        if ii.get("Dividends - equities"):      lines.append(f"  - Dividends (equities): {ii['Dividends - equities']}")
        if ii.get("Interest on cash deposits"): lines.append(f"  - Interest on cash deposits: {ii['Interest on cash deposits']}")
        if ii.get("Total"):                     lines.append(f"  - Total: {ii['Total']}")
        lines.append("")
    mf = N["Investment Management Costs"]
    if any(mf.values()):
        lines.append("Investment Management Costs:")
        if mf.get("Fees"): lines.append(f"  - Fees: {mf['Fees']}")
        lines.append("")
    ca = N["Costs of Charitable Activities"]
    if ca.get("lines") or ca.get("total"):
        lines.append("Costs of Charitable Activities:")
        for gl in ca.get("lines", []): lines.append(f"  - {gl}")
        if ca.get("total"):            lines.append(f"  - Total: {ca['total']}")
        lines.append("")
    im = N["Investments – Movement"]
    if any(im.values()):
        lines.append("Investments – Movement in Market Value:")
        order = ["Market Value at 1 April","Profits from Sales","Net loss on revaluations",
                 "Net Gain overall","Management Fees deducted","Market Value at 31 March"]
        for k in order:
            if im.get(k):
                lab = k
                if k == "Net Gain overall" and im[k].startswith("-"):
                    lab = "Net Loss overall"
                lines.append(f"  - {lab}: {im[k]}")
        lines.append("")
    db = N["Debtors"]
    if any(db.values()):
        lines.append("Debtors:")
        if db.get("Other Debtors"): lines.append(f"  - Other Debtors: {db['Other Debtors']}")
        lines.append("")
    cf = N["Analysis of Charitable Funds"]
    if cf.get("Closing Balance"):
        lines.append("Analysis of Charitable Funds:")
        lines.append(f"  - Closing Balance: {cf['Closing Balance']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"

def infer_year(full_text: str, filename: str) -> Optional[int]:
    m = re.search(r"YEAR ENDED 31 MARCH (\d{4})", full_text, flags=re.I)
    if m: return int(m.group(1))
    m2 = re.search(r"(20\d{2})", filename)
    return int(m2.group(1)) if m2 else None

def process_pdf(pdf_path: pathlib.Path, dpi: int = 300) -> Optional[pathlib.Path]:
    print(f"[extract] {pdf_path.name}")
    try:
        full_text = ocr_pdf_to_text(pdf_path, dpi=dpi)
    except Exception as e:
        print("ERROR running OCR. If on Windows, set POPPLER_PATH and TESSERACT_CMD env vars.\n"
              f"Details: {e}", file=sys.stderr)
        sys.exit(1)

    year = infer_year(full_text, pdf_path.name)
    period, bal_date = infer_period_and_date(full_text, pdf_path.name)
    parsed = parse_all(full_text)
    out_txt = format_txt(parsed, period, bal_date)
    out_path = OUT_DIR / output_name(pdf_path.name)

    if year == 2023 or year == 2024:
        # print("---- PREVIEW (2023) — not saving file as requested ----")
        print(f"  -> wrote {out_path}")
        # print(out_txt)
        return None  # do not save 2023
    else:
        out_path.write_text(out_txt, encoding="utf-8")
        print(f"  -> wrote {out_path}")
        return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=300, help="OCR DPI (default 300)")
    ap.add_argument("--poppler", type=str, default="", help="Poppler bin path (Windows)")
    ap.add_argument("--tesseract", type=str, default="", help="Tesseract binary path (Windows)")
    args = ap.parse_args()

    if args.poppler: os.environ["POPPLER_PATH"] = args.poppler
    if args.tesseract:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract

    pdfs = sorted(IN_DIR.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found in data/financial_reports_actual", file=sys.stderr)
        sys.exit(1)

    for pdf in pdfs:
        process_pdf(pdf, dpi=args.dpi)

if __name__ == "__main__":
    main()