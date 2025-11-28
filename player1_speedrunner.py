import argparse
import time
import urllib.parse
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer


WIKI_BASE = "https://en.wikipedia.org"
WIKI_SUMMARY_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"

HEADERS = {
    "User-Agent": "wikipedia-speedbot/0.1 (https://github.com/UniqueName54321; mailto:rndm1269@gmail.com)"
}



# ---------- utility stuff ----------

def title_from_url(url: str) -> str:
    """
    Convert a wiki URL to a nice title, e.g.
    https://en.wikipedia.org/wiki/Andrej_Karpathy -> Andrej Karpathy
    """
    last = urllib.parse.unquote(url.rsplit("/", 1)[-1])
    return last.replace("_", " ")


def normalize_wiki_url(url_or_title: str) -> str:
    """
    Normalize to a full wiki URL.
    """
    if url_or_title.startswith("http"):
        return url_or_title.split("#")[0]  # drop in-page anchors
    # treat as title
    return f"{WIKI_BASE}/wiki/{urllib.parse.quote(url_or_title.replace(' ', '_'))}"


def same_page(a: str, b: str) -> bool:
    """
    Compare two wiki URLs for 'same page' in a loose way.
    """
    a = normalize_wiki_url(a).lower()
    b = normalize_wiki_url(b).lower()
    return a == b


def fetch_page(url: str, delay: float = 0.0) -> str:
    """
    Fetch raw HTML for a page.
    """
    if delay > 0:
        time.sleep(delay)
    r = requests.get(url, timeout=20, headers=HEADERS)
    r.raise_for_status()
    return r.text


def extract_links(html: str) -> List[Dict[str, str]]:
    """
    Extract valid wiki article links from a page.
    Returns list of dicts: {"href": full_url, "text": visible_text}
    """
    soup = BeautifulSoup(html, "html.parser")
    soup.select("#mw-content-text a[href]")
    links = []
    for a in soup.select("a[href]"):
        href = a.get("href")
        text = (a.get_text() or "").strip()
        if not href or not text:
            continue

        # Only internal wiki article links like /wiki/Whatever
        if not href.startswith("/wiki/"):
            continue
        # Filter out non-article namespaces (File:, Help:, etc.)
        if any(href.startswith(prefix) for prefix in (
            "/wiki/Special:", "/wiki/File:", "/wiki/Help:",
            "/wiki/Talk:", "/wiki/Category:", "/wiki/Template:",
            "/wiki/Wikipedia:"
        )):
            continue
        if any(x in text.lower() for x in ["doi", "isbn", "jstor", "pmid", "issn", "hdl", "s2cid"]):
            continue
        import re
        if len(text) < 3 or not re.search(r"[a-zA-Z]", text):
            continue


        full = WIKI_BASE + href
        links.append({"href": full, "text": text})
    return links


def fetch_summary_for_target(url_or_title: str) -> str:
    """
    Use the REST API to get a short description of the target page.
    If it fails, fall back to just the title string.
    """
    title = title_from_url(normalize_wiki_url(url_or_title))
    api_url = WIKI_SUMMARY_API + urllib.parse.quote(title.replace(" ", "_"))
    try:
        r = requests.get(api_url, timeout=10, headers=HEADERS)
        if r.status_code != 200:
            return title
        j = r.json()
        parts = [
            j.get("title") or "",
            j.get("description") or "",
            j.get("extract") or "",
        ]
        text = ". ".join(p for p in parts if p).strip()
        return text or title
    except Exception:
        return title


# ---------- Player base + Player1_vectorized ----------

@dataclass
class Player:
    """
    Minimal base class. Only holds visualize_link so the Player1 code
    looks similar to the video, but here it's just pretty printing.
    """

    def visualize_link(
        self,
        link: Dict[str, str],
        page_url: str,
        step_index: int,
        highlight_color: Optional[str] = None,
        delay: float = 0.0,
    ):
        # simple console visualization
        tag = ""
        if highlight_color == "green":
            tag = "[TARGET?]"
        print(f"  - {link['text']}  ({link['href']}) {tag}")
        if delay > 0:
            time.sleep(delay)


class Player1Vectorized(Player):
    """
    Vectorized SBERT version of Player1.

    Uses a sentence-transformers model to embed:
      - the target description (single vector)
      - all link texts (batch)
    Then picks the link with minimal (1 - cosine_similarity).
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        # no device gymnastics; sentence-transformers handles it
        self.emb_model = SentenceTransformer(model_name)

    def __str__(self):
        return "Player1_vectorized"

    def choose_next_link(
        self,
        target_link: str,
        target_description: str,
        links: List[Dict[str, str]],
        visualize: bool,
        page_url: str,
        step_index: int,
    ) -> Dict[str, str]:
        clean_links: List[Dict[str, str]] = []
        clean_descriptions: List[str] = []

        for link in links:
            if visualize:
                self.visualize_link(link, page_url, step_index)

            # if href basically *is* the target, lock it in
            if same_page(target_link, link["href"]):
                if visualize:
                    self.visualize_link(
                        link,
                        page_url,
                        step_index,
                        highlight_color="green",
                        delay=0.3,
                    )
                return link

            if link["href"] != "" and link["text"] != "":
                clean_links.append(link)
                clean_descriptions.append(link["text"])

        if not clean_links:
            # super weird page, just fall back to first original link
            return links[0]

        # vectorized embedding
        target_emb = self.emb_model.encode(
            target_description,
            convert_to_tensor=False,
            normalize_embeddings=True,
        )  # shape (d,)

        link_emb = self.emb_model.encode(
            clean_descriptions,
            convert_to_tensor=False,
            normalize_embeddings=True,
        )  # shape (N, d)

        # cosine distance = 1 - cos_sim, since embeddings are normalized
        # link_emb @ target_emb -> (N,) dot products
        dots = link_emb @ target_emb
        dis = 1.0 - dots

        order = np.argsort(dis)
        best = clean_links[int(order[0])]

        if visualize:
            print("\nChosen link:")
            self.visualize_link(
                best,
                page_url,
                step_index,
                highlight_color="green",
                delay=0.1,
            )
            print()

        return best


# ---------- main loop / CLI ----------

def run_speedrun(
    start_url: str,
    target_url: str,
    target_description: Optional[str],
    max_steps: int,
    visualize: bool,
):
    player = Player1Vectorized()

    start_url = normalize_wiki_url(start_url)
    target_url = normalize_wiki_url(target_url)

    if not target_description:
        print("[info] No description provided, fetching from Wikipedia summary API…")
        target_description = fetch_summary_for_target(target_url)
    print(f"[target description] {target_description}\n")

    current = start_url
    path = [current]

    t0 = time.time()
    for step in range(max_steps):
        print(f"Step {step} — current page: {title_from_url(current)}")
        if same_page(current, target_url):
            print("Already at target!")
            break

        html = fetch_page(current)
        links = extract_links(html)

        if not links:
            print("No outgoing links found, dead end.")
            break

        next_link = player.choose_next_link(
            target_link=target_url,
            target_description=target_description,
            links=links,
            visualize=visualize,
            page_url=current,
            step_index=step,
        )

        current = next_link["href"]
        path.append(current)

        if same_page(current, target_url):
            print(f"Reached target page: {title_from_url(current)}")
            break

        print()

    t1 = time.time()

    print("\n=== RUN COMPLETE ===")
    print(f"Time: {t1 - t0:.3f}s")
    print(f"Steps taken: {len(path) - 1}")
    print("Path:")
    for i, url in enumerate(path):
        print(f"  {i:2d}: {title_from_url(url)}  ({url})")


def main():
    ap = argparse.ArgumentParser(description="Player1 SBERT Wikipedia speedrunner")
    ap.add_argument("--start", required=True, help="Start page (title or full URL)")
    ap.add_argument("--target", required=True, help="Target page (title or full URL)")
    ap.add_argument(
        "--desc",
        default=None,
        help="Optional natural-language description of the target page "
             "(otherwise we use Wikipedia's summary API)",
    )
    ap.add_argument("--max-steps", type=int, default=10, help="Max hops before giving up")
    ap.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable printing all candidate links each step",
    )

    args = ap.parse_args()

    run_speedrun(
        start_url=args.start,
        target_url=args.target,
        target_description=args.desc,
        max_steps=args.max_steps,
        visualize=not args.no_viz,
    )


if __name__ == "__main__":
    main()
