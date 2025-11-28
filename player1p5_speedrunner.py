import argparse
import time
import urllib.parse
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple, Union

import re
import numpy as np
import requests
from bs4 import BeautifulSoup


# -------------------------------------------------------------------
# SBERT HTTP CLIENT (REQUIRED)
# -------------------------------------------------------------------

class SBERTHTTPClient:
    """Simple HTTP client for the external SBERT server."""

    def __init__(self, base_url: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Send text(s) to the server and receive embeddings."""
        single = False
        if isinstance(texts, str):
            texts = [texts]
            single = True

        payload = {
            "texts": texts,
            "normalize": True,
        }

        try:
            r = self.session.post(
                self.base_url + "/embed",
                json=payload,
                timeout=self.timeout,
            )
            r.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f"Could not reach SBERT server at {self.base_url}. Error: {e}"
            )

        arr = np.asarray(r.json()["embeddings"], dtype=np.float32)
        return arr[0] if single else arr


# -------------------------------------------------------------------
# WIKIPEDIA HELPERS
# -------------------------------------------------------------------

WIKI_BASE = "https://en.wikipedia.org"
WIKI_SUMMARY_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"

HEADERS = {
    "User-Agent": "wiki-speedrunner/1.0"
}


def title_from_url(url: str) -> str:
    last = urllib.parse.unquote(url.rsplit("/", 1)[-1])
    return last.replace("_", " ")


def normalize_wiki_url(url_or_title: str) -> str:
    if url_or_title.startswith("http"):
        return url_or_title.split("#")[0]
    return f"{WIKI_BASE}/wiki/{urllib.parse.quote(url_or_title.replace(' ', '_'))}"


def same_page(a: str, b: str) -> bool:
    return normalize_wiki_url(a).lower() == normalize_wiki_url(b).lower()


def fetch_page(url: str) -> str:
    r = requests.get(url, timeout=20, headers=HEADERS)
    r.raise_for_status()
    return r.text


def extract_links(html: str, allow_non_mainspace: bool = False) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    content = soup.select_one("#mw-content-text")
    anchors = content.select("a[href]") if content else soup.select("a[href]")

    links = []
    non_main = (
        "Special:", "File:", "Help:", "Talk:", "Category:", "Template:",
        "Wikipedia:", "Portal:", "Draft:"
    )

    for a in anchors:
        href = a.get("href")
        text = (a.get_text() or "").strip()
        if not href or not text or href.startswith("#"):
            continue

        rel = href.startswith("/wiki/")
        abs_int = href.startswith(WIKI_BASE + "/wiki/")
        if not (rel or abs_int):
            continue

        full = WIKI_BASE + href if rel else href
        title_part = full.split("/wiki/", 1)[1] if "/wiki/" in full else ""

        is_non = any(title_part.startswith(prefix) for prefix in non_main)
        if is_non and not allow_non_mainspace:
            continue

        if any(x in text.lower() for x in ["doi", "isbn", "pmid", "issn", "jstor", "hdl"]):
            continue

        if len(text) < 3 or not re.search(r"[a-zA-Z]", text):
            continue

        links.append({"href": full, "text": text, "non_mainspace": is_non})

    return links


def fetch_summary_for_target(url_or_title: str) -> str:
    title = title_from_url(normalize_wiki_url(url_or_title))
    api_url = WIKI_SUMMARY_API + urllib.parse.quote(title.replace(" ", "_"))

    try:
        r = requests.get(api_url, timeout=10, headers=HEADERS)
        if r.status_code != 200:
            return title
        j = r.json()
        parts = [j.get("title") or "", j.get("description") or "", j.get("extract") or ""]
        return ". ".join(p for p in parts if p).strip() or title
    except:
        return title


# -------------------------------------------------------------------
# PLAYER BASE
# -------------------------------------------------------------------

@dataclass
class Player:
    """Base class with simple visualization printing."""

    def visualize_link(self, link: Dict[str, str], page_url: str, step: int,
                       highlight_color: Optional[str] = None,
                       delay: float = 0.0, is_candidate: bool = False):
        tags = []
        if is_candidate: tags.append("[CAND]")
        if highlight_color == "green": tags.append("[CHOSEN]")
        if link.get("non_mainspace"): tags.append("[NON-MAIN]")
        tag = " ".join(tags)
        print(f"  - {link['text']}  ({link['href']}) {tag}")
        if delay > 0:
            time.sleep(delay)


# -------------------------------------------------------------------
# PLAYER 1.5 — SBERT SERVER ONLY
# -------------------------------------------------------------------

class Player1p5Vectorized(Player):
    """Player1.5 using remote SBERT server ONLY."""

    def __init__(self, sbert_url: str, summary_weight: float = 0.3):
        super().__init__()
        self.client = SBERTHTTPClient(sbert_url)
        self.summary_weight = summary_weight

        self._target_cache: Dict[str, np.ndarray] = {}
        self._summary_cache: Dict[str, Tuple[str, np.ndarray]] = {}

    def __str__(self):
        return "Player1.5_HTTP"

    def _embed_target(self, text: str) -> np.ndarray:
        text = text.strip()
        if text not in self._target_cache:
            self._target_cache[text] = self.client.encode(text)
        return self._target_cache[text]

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        return np.zeros((0, 0), dtype=np.float32) if not texts else self.client.encode(texts)

    def _embed_summary(self, href: str) -> Tuple[str, np.ndarray]:
        key = normalize_wiki_url(href)
        if key in self._summary_cache:
            return self._summary_cache[key]
        summary = fetch_summary_for_target(key)
        emb = self.client.encode(summary)
        self._summary_cache[key] = (summary, emb)
        return summary, emb

    def _cosine_dist(self, M: np.ndarray, t: np.ndarray) -> np.ndarray:
        if M.size == 0:
            return np.zeros((0,), dtype=np.float32)
        return 1.0 - (M @ t)

    def choose_next_link(self, target_url: str, target_desc: str, links, visualize,
                         page_url, step, visited, beam_width, **kwargs):

        target_emb = self._embed_target(target_desc)
        target_norm = normalize_wiki_url(target_url)

        clean_links = []
        clean_texts = []

        for link in links:
            href = normalize_wiki_url(link["href"])
            if href in visited:
                continue

            if same_page(target_norm, href):
                if visualize:
                    self.visualize_link(link, page_url, step, "green", 0.2)
                return link

            clean_links.append(link)
            clean_texts.append(link["text"])

        if not clean_links:
            for link in links:
                if normalize_wiki_url(link["href"]) not in visited:
                    return link
            return links[0]

        anchor_emb = self._embed_texts(clean_texts)
        anchor_dis = self._cosine_dist(anchor_emb, target_emb)

        order = np.argsort(anchor_dis)
        beam_idxs = list(order[: min(beam_width, len(order))])

        if beam_width <= 1:
            return clean_links[beam_idxs[0]]

        cand_scores = []
        for idx in beam_idxs:
            link = clean_links[idx]
            _, sum_emb = self._embed_summary(link["href"])
            sum_dis = self._cosine_dist(sum_emb.reshape(1, -1), target_emb)[0]
            combined = (1 - self.summary_weight) * float(anchor_dis[idx]) + \
                       self.summary_weight * float(sum_dis)
            cand_scores.append((combined, link))

        cand_scores.sort(key=lambda x: x[0])
        return cand_scores[0][1]


# -------------------------------------------------------------------
# MAIN SPEEDRUN LOOP
# -------------------------------------------------------------------

def run_speedrun(start_url, target_url, target_desc, max_steps, visualize,
                 beam_width, allow_non_mainspace, sbert_url):

    player = Player1p5Vectorized(sbert_url=sbert_url)

    start_url = normalize_wiki_url(start_url)
    target_url = normalize_wiki_url(target_url)

    if not target_desc:
        print("[info] Fetching target description…")
        target_desc = fetch_summary_for_target(target_url)

    target_title = title_from_url(target_url)
    print(f"[target] {target_title} ({target_url})")
    print(f"[target description] {target_desc}\n")

    current = start_url
    path = [current]
    visited = {current}

    t0 = time.time()

    for step in range(max_steps):
        print(f"Step {step} — current: {title_from_url(current)}")
        print(f"  Target: {target_title}\n")

        if same_page(current, target_url):
            print("Already at target!")
            break

        html = fetch_page(current)
        links = extract_links(html, allow_non_mainspace)

        if not links:
            print("Dead end.")
            break

        next_link = player.choose_next_link(
            target_url=target_url,
            target_desc=target_desc,
            links=links,
            visualize=visualize,
            page_url=current,
            step=step,
            visited=visited,
            beam_width=beam_width,
        )

        current = normalize_wiki_url(next_link["href"])

        if current in visited:
            print("Loop detected, aborting.")
            path.append(current)
            break

        visited.add(current)
        path.append(current)

        if same_page(current, target_url):
            print(f"Reached target page: {title_from_url(current)}")
            break

        print()

    t1 = time.time()

    print("\n=== RUN COMPLETE ===")
    print(f"Time: {t1 - t0:.3f}s")
    print(f"Steps: {len(path) - 1}")
    for i, url in enumerate(path):
        print(f"{i:2d}: {title_from_url(url)} ({url})")


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Player1.5 SBERT-speed Wikipedia runner")
    ap.add_argument("--start", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--desc", default=None)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--no-viz", action="store_true")
    ap.add_argument("--beam-width", type=int, default=3)
    ap.add_argument("--allow-non-mainspace", action="store_true")
    ap.add_argument("--sbert-url", required=True,
                    help="URL of SBERT embedding server (required)")

    args = ap.parse_args()

    run_speedrun(
        start_url=args.start,
        target_url=args.target,
        target_desc=args.desc,
        max_steps=args.max_steps,
        visualize=not args.no_viz,
        beam_width=args.beam_width,
        allow_non_mainspace=args.allow_non_mainspace,
        sbert_url=args.sbert_url,
    )


if __name__ == "__main__":
    main()
