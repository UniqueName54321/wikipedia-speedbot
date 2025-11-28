import argparse
import time
import urllib.parse
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple

import re
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


def extract_links(
    html: str,
    allow_non_mainspace: bool = False,
) -> List[Dict[str, str]]:
    """
    Extract links from a page.

    - Always stay inside Wikipedia (/wiki/...).
    - If allow_non_mainspace is False:
        * only keep main article namespace (no Category:, Help:, etc.)
    - If allow_non_mainspace is True:
        * include those, and mark them as non_mainspace=True
    """
    soup = BeautifulSoup(html, "html.parser")

    # Prefer main content area if it exists
    content = soup.select_one("#mw-content-text")
    if content is not None:
        anchors = content.select("a[href]")
    else:
        anchors = soup.select("a[href]")

    links: List[Dict[str, str]] = []

    for a in anchors:
        href = a.get("href")
        text = (a.get_text() or "").strip()
        if not href or not text:
            continue

        # Skip pure in-page anchors like "#Section"
        if href.startswith("#"):
            continue

        # We only care about internal wiki links
        is_relative_internal = href.startswith("/wiki/")
        is_absolute_internal = href.startswith(WIKI_BASE + "/wiki/")
        if not (is_relative_internal or is_absolute_internal):
            # drop external / mailto / javascript / etc.
            continue

        # Normalize internal URLs to full form
        if is_relative_internal:
            full = WIKI_BASE + href
        else:
            full = href

        # Extract the title portion after /wiki/
        try:
            title_part = full.split("/wiki/", 1)[1]
        except IndexError:
            title_part = ""

        # Determine if this is a non-mainspace page (Category:, File:, etc.)
        non_mainspace_prefixes = (
            "Special:", "File:", "Help:", "Talk:",
            "Category:", "Template:", "Wikipedia:",
            "Portal:", "Draft:",
        )
        is_non_mainspace = any(title_part.startswith(prefix) for prefix in non_mainspace_prefixes)

        # If we're NOT allowing non-mainspace, filter those out
        if is_non_mainspace and not allow_non_mainspace:
            continue

        # Filter out obvious metadata cruft by text
        if any(x in text.lower() for x in ["doi", "isbn", "jstor", "pmid", "issn", "hdl", "s2cid"]):
            continue

        # Require somewhat normal-looking anchor text
        if len(text) < 3 or not re.search(r"[a-zA-Z]", text):
            continue

        links.append({
            "href": full,
            "text": text,
            "non_mainspace": is_non_mainspace,
        })

    return links




def fetch_summary_for_target(url_or_title: str) -> str:
    """
    Use the REST API to get a short description of a page.
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


# ---------- Player base + Player1.5 (vectorized + tiny beam) ----------

@dataclass
class Player:
    """
    Minimal base class. Only holds visualize_link so the Player code
    looks similar to the video, but here it's just pretty printing.
    """

    def visualize_link(
        self,
        link: Dict[str, str],
        page_url: str,
        step_index: int,
        highlight_color: Optional[str] = None,
        delay: float = 0.0,
        is_candidate: bool = False,
    ):
        # simple console visualization
        tag_parts = []
        if is_candidate:
            tag_parts.append("[CAND]")
        if highlight_color == "green":
            tag_parts.append("[CHOSEN]")
        if link.get("non_mainspace"):
            tag_parts.append("[NON-MAIN]")  # <--- new

        tag = " ".join(tag_parts)
        print(f"  - {link['text']}  ({link['href']}) {tag}")
        if delay > 0:
            time.sleep(delay)


class Player1p5Vectorized(Player):
    """
    Player1.5 (clean version):
      - SBERT-based link chooser
      - Vectorized anchor-text scoring (like Player1)
      - Tiny beam over top-K links using page summaries:
        * Score links by similarity of anchor text to target description
        * Take top beam_width links
        * For those, look at page summary and rescore
      - No non-mainspace bias, no substring 'cat' hacks
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.emb_model = SentenceTransformer(model_name)

    def __str__(self):
        return "Player1.5_vectorized"

    # --- core scoring helpers ---

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        emb = self.emb_model.encode(
            texts,
            convert_to_tensor=False,
            normalize_embeddings=True,
        )
        return np.asarray(emb)

    def _cosine_distances(self, emb_matrix: np.ndarray, emb_target: np.ndarray) -> np.ndarray:
        """
        emb_matrix: (N, d)
        emb_target: (d,)
        returns 1 - cosine_similarity for each row.
        """
        dots = emb_matrix @ emb_target  # (N,)
        return 1.0 - dots

    def choose_next_link(
        self,
        target_link: str,
        target_description: str,
        links: List[Dict[str, str]],
        visualize: bool,
        page_url: str,
        step_index: int,
        visited_urls: Set[str],
        beam_width: int = 3,
        prioritize_non_mainspace: bool = False,  # kept for compatibility, not used
    ) -> Dict[str, str]:

        """
        Main policy:
          1) Filter links to unvisited & non-empty.
          2) If any is literally the target page, snap to it.
          3) Embed all remaining anchor texts; sort by distance to target_desc.
          4) Take top-K candidates (beam_width).
          5) For those K, fetch page summaries, embed, and combine scores:
             score = (1 - summary_weight) * anchor_distance + summary_weight * summary_distance
          6) Pick min score.
        """
        clean_links: List[Dict[str, str]] = []
        clean_texts: List[str] = []

        target_norm = normalize_wiki_url(target_link)

        # Step 1 + 2: filter & instant target match
        for link in links:
            if link.get("external"):
                href_norm = link["href"]
            else:
                href_norm = normalize_wiki_url(link["href"])

            # skip already visited pages
            if href_norm in visited_urls:
                continue

            if link["href"] != "" and link["text"] != "":
                # exact/loose target match
                if same_page(target_norm, href_norm):
                    if visualize:
                        self.visualize_link(
                            link,
                            page_url,
                            step_index,
                            highlight_color="green",
                            delay=0.2,
                        )
                    return link

                clean_links.append(link)
                clean_texts.append(link["text"])

        # fallback if everything got filtered out
        if not clean_links:
            # just pick the first non-visited original link, or first of all
            for link in links:
                href_norm = normalize_wiki_url(link["href"])
                if href_norm not in visited_urls:
                    return link
            return links[0]

        # --- Step 3: anchor-text scoring (like Player1) ---

        # Single target embedding from description, like original Player1
        target_emb = self.emb_model.encode(
            target_description,
            convert_to_tensor=False,
            normalize_embeddings=True,
        )

        anchor_emb = self._encode_texts(clean_texts)
        anchor_dis = self._cosine_distances(anchor_emb, target_emb)

        # sort by anchor distance (lower is better)
        order = np.argsort(anchor_dis)
        beam_indices = list(order[: min(beam_width, len(order))])

        # If beam_width == 1, just classic greedy (Player1 behavior)
        if beam_width <= 1 and beam_indices:
            best_idx = int(beam_indices[0])
            best_link = clean_links[best_idx]
            if visualize:
                print("\nChosen link:")
                self.visualize_link(
                    best_link,
                    page_url,
                    step_index,
                    highlight_color="green",
                    delay=0.1,
                )
                print()
            return best_link

        # --- Step 4–5: re-score top-K using page summaries (small tweak) ---

        summary_weight = 0.3  # keep this modest so anchor text still dominates

        candidate_scores: List[Tuple[float, Dict[str, str]]] = []
        for idx in beam_indices:
            idx = int(idx)
            link = clean_links[idx]
            href = link["href"]

            # Fetch summary for that page
            summary = fetch_summary_for_target(href)

            summary_emb = self.emb_model.encode(
                summary,
                convert_to_tensor=False,
                normalize_embeddings=True,
            )
            summary_dis = 1.0 - float(summary_emb @ target_emb)

            # combine anchor + summary distances
            combined = (1.0 - summary_weight) * float(anchor_dis[idx]) + summary_weight * summary_dis

            candidate_scores.append((combined, link))

        candidate_scores.sort(key=lambda x: x[0])
        best_score, best_link = candidate_scores[0]

        if visualize:
            print("\nTop candidates this step (after beam re-score):")
            for score, link in candidate_scores:
                self.visualize_link(
                    link,
                    page_url,
                    step_index,
                    is_candidate=True,
                )
                print(f"    -> combined distance: {score:.4f}")
            print("\nChosen link:")
            self.visualize_link(
                best_link,
                page_url,
                step_index,
                highlight_color="green",
                delay=0.1,
            )
            print()

        return best_link





# ---------- main loop / CLI ----------

def run_speedrun(
    start_url: str,
    target_url: str,
    target_description: Optional[str],
    max_steps: int,
    visualize: bool,
    beam_width: int,
    allow_non_mainspace: bool,
):
    player = Player1p5Vectorized()

    start_url = normalize_wiki_url(start_url)
    target_url = normalize_wiki_url(target_url)

    if not target_description:
        print("[info] No description provided, fetching from Wikipedia summary API…")
        target_description = fetch_summary_for_target(target_url)

    target_title = title_from_url(target_url)
    print(f"[target page] {target_title}  ({target_url})")
    print(f"[target description] {target_description}\n")

    current = start_url
    path = [current]
    visited: Set[str] = {normalize_wiki_url(current)}

    t0 = time.time()
    for step in range(max_steps):
        print(f"Step {step} — current page: {title_from_url(current)}")
        print(f"  Target: {target_title}\n")

        if same_page(current, target_url):
            print("Already at target!")
            break

        html = fetch_page(current)
        links = extract_links(
            html,
            allow_non_mainspace=allow_non_mainspace,
        )

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
            visited_urls=visited,
            beam_width=beam_width,
            prioritize_non_mainspace=allow_non_mainspace,  # <--
        )

        if next_link.get("external"):
            current = next_link["href"]
        else:
            current = normalize_wiki_url(next_link["href"])
        
        if current in visited:
            print("Stuck in a loop / revisit, bailing out.")
            path.append(current)
            break

        visited.add(current)
        path.append(current)

        if same_page(current, target_url):
            print(f"Reached target page: {title_from_url(current)}")
            break

        print()

    t1 = time.time()

    print("\n=== RUN COMPLETE (Player1.5) ===")
    print(f"Time: {t1 - t0:.3f}s")
    print(f"Steps taken: {len(path) - 1}")
    print("Path:")
    for i, url in enumerate(path):
        print(f"  {i:2d}: {title_from_url(url)}  ({url})")


def main():
    ap = argparse.ArgumentParser(description="Player1.5 SBERT Wikipedia speedrunner")
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
    ap.add_argument(
        "--beam-width",
        type=int,
        default=3,
        help="How many top candidates to consider in the tiny beam (default: 3)",
    )
    ap.add_argument(
        "--allow-non-mainspace",
        action="store_true",
        help="Allow following links into non-article namespaces (File:, Help:, Category:, etc.)",
    )

    args = ap.parse_args()

    run_speedrun(
        start_url=args.start,
        target_url=args.target,
        target_description=args.desc,
        max_steps=args.max_steps,
        visualize=not args.no_viz,
        beam_width=args.beam_width,
        allow_non_mainspace=args.allow_non_mainspace,
    )



if __name__ == "__main__":
    main()
