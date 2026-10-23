#!/usr/bin/env python3
"""
Generic Scientific Paper Miner
------------------------------
Multi-source search with live config editor in GUI
"""

import json
import re
import threading
import time
import tkinter as tk
from datetime import datetime, timedelta
from tkinter import filedialog, messagebox, scrolledtext, ttk

import arxiv
import pandas as pd
import requests
from Bio import Entrez, Medline

# ============== DEFAULT CONFIG ==============
DEFAULT_EMAIL = "your_email@example.com"
MAX_PAPERS_PER_SOURCE = 20
DAYS_BACK = 365

METHOD_KEYWORDS = {
    "experimental": ["experiment", "experimental"],
    "computational": ["simulation", "modeling", "computational"],
    "survey": ["survey", "questionnaire"],
    "statistical": ["regression", "anova", "bayesian"],
}

EFFECT_PATTERNS = [
    (r"cohen'?s?\s*d\s*[=:]\s*([\d.]+)", "cohens_d"),
    (r"r\s*[=:]\s*([\d.]+)", "correlation_r"),
    (r"beta\s*[=:]\s*([\d.]+)", "beta"),
    (r"(?:η²|eta.?squared?)\s*[=:]\s*([\d.]+)", "eta_squared"),
    (r"effect size\s*[=:]\s*([\d.]+)", "effect_size"),
]

# Global config (modified via GUI)
config = {
    "email": DEFAULT_EMAIL,
    "method_keywords": METHOD_KEYWORDS.copy(),
    "effect_patterns": EFFECT_PATTERNS.copy(),
}


# ============== UTILITIES ==============
def safe_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def clean_text(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def detect_methods(text):
    text_lower = text.lower()
    detected = []
    for method, kws in config["method_keywords"].items():
        if any(kw in text_lower for kw in kws):
            detected.append(method)
    return list(set(detected))


def extract_entities(text):
    result = {"entity": "", "controls": "", "value": ""}
    if not text:
        return result
    entity_match = re.search(
        r"\b([A-Za-z]+ (disease|syndrome|model|effect|group|treatment))\b", text, re.I
    )
    if entity_match:
        result["entity"] = entity_match.group(1)
    if re.search(r"control (group|condition)", text, re.I):
        result["controls"] = "yes"
    value_match = re.search(r"(?:mean|average)[:\s]*([\d\-.]+)", text, re.I)
    if value_match:
        result["value"] = value_match.group(1)
    return result


def extract_effect_sizes(text):
    effects = []
    for pat, etype in config["effect_patterns"]:
        for m in re.findall(pat, text, re.I):
            val = safe_float(m)
            if val is not None:
                effects.append({"type": etype, "value": val})
    return effects


# ============== SEARCH FUNCTIONS (unchanged) ==============
def search_pubmed(keyword, max_results=MAX_PAPERS_PER_SOURCE):
    Entrez.email = config["email"]
    cutoff = (datetime.now() - timedelta(days=DAYS_BACK)).strftime("%Y/%m/%d")
    query = f'("{keyword}"[Title/Abstract]) AND ("{cutoff}"[Date - Publication] : "3000"[Date - Publication])'
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    pmids = record.get("IdList", [])
    if not pmids:
        return []
    handle = Entrez.efetch(
        db="pubmed", id=",".join(pmids), rettype="medline", retmode="text"
    )
    records = Medline.parse(handle)
    papers = []
    for rec in records:
        papers.append(
            {
                "source": "pubmed",
                "id": rec.get("PMID", ""),
                "title": rec.get("TI", ""),
                "abstract": rec.get("AB", ""),
                "date": rec.get("DP", ""),
                "journal": rec.get("TA", ""),
            }
        )
    handle.close()
    return papers


def search_arxiv(keyword, max_results=MAX_PAPERS_PER_SOURCE):
    client = arxiv.Client()
    search = arxiv.Search(
        query=keyword,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    papers = []
    for r in client.results(search):
        papers.append(
            {
                "source": "arxiv",
                "id": r.get_short_id(),
                "title": r.title,
                "abstract": r.summary,
                "date": r.published.strftime("%Y-%m-%d"),
                "journal": "arXiv",
                "url": r.entry_id,
            }
        )
    return papers


def search_biorxiv(keyword, max_results=MAX_PAPERS_PER_SOURCE):
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d")
    papers = []
    cursor = 0
    max_pages = (
        100  # bioRxiv has no keyword search; cap client-side pagination for rare terms
    )
    for _ in range(max_pages):
        if len(papers) >= max_results:
            break
        url = (
            f"https://api.biorxiv.org/details/biorxiv/{start_date}/{end_date}/{cursor}"
        )
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                break
            data = resp.json()
            collection = data.get("collection", [])
            if not collection:
                break
            for item in collection:
                full_text = (
                    (item.get("title") or "") + " " + (item.get("abstract") or "")
                )
                if keyword.lower() in full_text.lower():
                    papers.append(
                        {
                            "source": "biorxiv",
                            "id": item.get("doi"),
                            "title": item.get("title"),
                            "abstract": item.get("abstract"),
                            "date": item.get("date"),
                            "journal": "bioRxiv",
                            "url": item.get("url"),
                        }
                    )
                    if len(papers) >= max_results:
                        break
            cursor += len(collection)
            time.sleep(0.3)
        except Exception:
            break
    return papers[:max_results]


def _reconstruct_abstract(inverted_index):
    if not inverted_index:
        return ""
    positions = {}
    for word, idxs in inverted_index.items():
        for idx in idxs:
            positions[idx] = word
    return " ".join(positions[i] for i in sorted(positions))


def search_openalex(keyword, max_results=MAX_PAPERS_PER_SOURCE):
    papers = []
    url = f"https://api.openalex.org/works?search={keyword}&per_page={max_results}&sort=publication_date:desc"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            for item in data.get("results", []):
                abstract = (
                    _reconstruct_abstract(item.get("abstract_inverted_index"))
                    or "No abstract"
                )
                source = (item.get("primary_location") or {}).get("source") or {}
                papers.append(
                    {
                        "source": "openalex",
                        "id": item.get("id"),
                        "title": item.get("title"),
                        "abstract": abstract,
                        "date": item.get("publication_date"),
                        "journal": source.get("display_name", "Unknown"),
                        "url": (item.get("ids") or {}).get("doi") or item.get("id"),
                    }
                )
    except Exception:
        pass
    return papers


def process_paper(paper):
    text = clean_text(paper.get("title", "") + ". " + paper.get("abstract", ""))
    methods = detect_methods(text)
    entities = extract_entities(text)
    effects = extract_effect_sizes(text)
    effect_str = "; ".join([f"{e['type']}={e['value']:.2f}" for e in effects])

    return {
        **paper,
        "methods": ", ".join(methods) if methods else "none",
        "entity": entities["entity"],
        "controls": entities["controls"],
        "value": entities["value"],
        "effect_sizes": effect_str,
        "n_effects": len(effects),
    }


# ============== GUI ==============
class PaperMinerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Scientific Paper Miner - Configurable")
        self.root.geometry("1180x820")

        # Top controls
        top_frame = ttk.Frame(root, padding=10)
        top_frame.pack(fill="x")

        ttk.Label(top_frame, text="Search Term:").pack(side="left")
        self.keyword_var = tk.StringVar(value="machine learning")
        self.entry = ttk.Entry(top_frame, textvariable=self.keyword_var, width=40)
        self.entry.pack(side="left", padx=8)

        self.search_btn = ttk.Button(
            top_frame, text="Search All Sources", command=self.start_search
        )
        self.search_btn.pack(side="left", padx=5)

        ttk.Button(top_frame, text="Clear", command=self.clear_output).pack(
            side="left", padx=5
        )

        # Config buttons
        ttk.Button(top_frame, text="Set Email", command=self.edit_email).pack(
            side="left", padx=5
        )
        ttk.Button(top_frame, text="Edit Methods", command=self.edit_methods).pack(
            side="left", padx=5
        )
        ttk.Button(
            top_frame, text="Edit Effect Patterns", command=self.edit_effect_patterns
        ).pack(side="left", padx=5)

        self.progress = ttk.Progressbar(top_frame, mode="indeterminate")
        self.progress.pack(side="left", fill="x", expand=True, padx=10)

        # Export
        export_frame = ttk.Frame(root, padding=(10, 2))
        export_frame.pack(fill="x")
        ttk.Button(export_frame, text="Export CSV", command=self.export_csv).pack(
            side="left", padx=5
        )
        ttk.Button(export_frame, text="Export JSON", command=self.export_json).pack(
            side="left", padx=5
        )

        # Results
        self.result_text = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, font=("Consolas", 10)
        )
        self.result_text.pack(fill="both", expand=True, padx=10, pady=8)

        self.status_var = tk.StringVar(value=f"Ready | Email: {config['email']}")
        ttk.Label(root, textvariable=self.status_var).pack(side="bottom", pady=5)

        self.df = None

    def edit_email(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Set NCBI Email")
        dialog.geometry("400x150")
        ttk.Label(dialog, text="Enter your email (required for PubMed):").pack(pady=10)
        email_var = tk.StringVar(value=config["email"])
        entry = ttk.Entry(dialog, textvariable=email_var, width=50)
        entry.pack(pady=5)

        def save():
            config["email"] = email_var.get().strip()
            self.status_var.set(f"Ready | Email: {config['email']}")
            dialog.destroy()

        ttk.Button(dialog, text="Save", command=save).pack(pady=10)

    def edit_methods(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Edit Method Keywords")
        dialog.geometry("600x500")
        text = tk.Text(dialog, height=20, width=70)
        text.pack(pady=10, padx=10)
        text.insert(tk.END, json.dumps(config["method_keywords"], indent=2))

        def save():
            try:
                new_dict = json.loads(text.get("1.0", tk.END))
                config["method_keywords"] = new_dict
                messagebox.showinfo("Success", "Method keywords updated!")
                dialog.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Invalid JSON: {e}")

        ttk.Button(dialog, text="Save Changes", command=save).pack(pady=5)

    def edit_effect_patterns(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Edit Effect Size Patterns")
        dialog.geometry("700x500")
        text = tk.Text(dialog, height=20, width=80)
        text.pack(pady=10, padx=10)
        text.insert(tk.END, json.dumps(config["effect_patterns"], indent=2))

        def save():
            try:
                new_list = json.loads(text.get("1.0", tk.END))
                config["effect_patterns"] = new_list
                messagebox.showinfo("Success", "Effect patterns updated!")
                dialog.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Invalid JSON: {e}")

        ttk.Button(dialog, text="Save Changes", command=save).pack(pady=5)

    def start_search(self):
        keyword = self.keyword_var.get().strip()
        if not keyword:
            messagebox.showwarning("Input Required", "Please enter a search term")
            return
        if config["email"] == DEFAULT_EMAIL or "@" not in config["email"]:
            messagebox.showwarning(
                "Email Required",
                "Set a real email via 'Set Email' first (required by NCBI/PubMed).",
            )
            return

        self.search_btn.config(state="disabled")
        self.progress.start()
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"Searching '{keyword}'...\n\n")
        self.status_var.set("Searching across 4 sources...")

        threading.Thread(target=self.run_search, args=(keyword,), daemon=True).start()

    def run_search(self, keyword):
        try:
            all_papers = []
            for label, fn in [
                ("PubMed", search_pubmed),
                ("arXiv", search_arxiv),
                ("bioRxiv", search_biorxiv),
                ("OpenAlex", search_openalex),
            ]:
                self.update_output(f"→ {label}...\n")
                try:
                    results = fn(keyword)
                    all_papers.extend(results)
                except Exception as e:
                    self.update_output(f"   ⚠ {label} failed: {e}\n")

            self.update_output(f"\nTotal: {len(all_papers)} papers. Extracting...\n\n")

            processed = [process_paper(p) for p in all_papers]
            self.df = pd.DataFrame(processed)
            self.show_summary()
        except Exception as e:
            self.update_output(f"\nError: {str(e)}\n")
        finally:
            self.root.after(0, self.finish_search)

    def update_output(self, text):
        self.root.after(0, lambda: self.result_text.insert(tk.END, text))

    def show_summary(self):
        if self.df is None or self.df.empty:
            self.update_output("No results.\n")
            return
        # (same summary code as before)
        summary = (
            "\n"
            + "=" * 80
            + "\nRESULTS SUMMARY\n"
            + "=" * 80
            + f"\nTotal papers: {len(self.df)}\n"
        )
        summary += f"Sources: {self.df['source'].value_counts().to_dict()}\n\n"
        # Methods
        method_counts = {}
        for m in self.df["methods"]:
            for meth in str(m).split(", "):
                if meth != "none":
                    method_counts[meth] = method_counts.get(meth, 0) + 1
        summary += "--- Methods ---\n"
        for m, c in sorted(method_counts.items(), key=lambda x: -x[1]):
            summary += f"• {m}: {c}\n"
        # Entities & Top papers (shortened for brevity)
        summary += "\n--- Top Entities ---\n"
        for ent, c in self.df["entity"].value_counts().head(8).items():
            if ent:
                summary += f"• {ent}: {c}\n"
        self.update_output(summary)
        self.update_output("\n✅ Use Export buttons above\n")

    def finish_search(self):
        self.progress.stop()
        self.search_btn.config(state="normal")
        self.status_var.set(f"Ready | Email: {config['email']}")

    def clear_output(self):
        self.result_text.delete(1.0, tk.END)

    def export_csv(self):
        if self.df is None or self.df.empty:
            messagebox.showwarning("No Data", "Run a search first")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")]
        )
        if path:
            self.df.to_csv(path, index=False)
            messagebox.showinfo("Success", f"Saved: {path}")

    def export_json(self):
        if self.df is None or self.df.empty:
            messagebox.showwarning("No Data", "Run a search first")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON", "*.json")]
        )
        if path:
            self.df.to_json(path, orient="records", indent=2)
            messagebox.showinfo("Success", f"Saved: {path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = PaperMinerGUI(root)
    root.mainloop()
