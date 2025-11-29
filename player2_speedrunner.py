import argparse
import os
import time
import urllib.parse
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple

import re
import requests
import numpy as np
from bs4 import BeautifulSoup

import json
from collections import Counter



# -------------------------------------------------------------------
# WIKIPEDIA HELPERS
# -------------------------------------------------------------------

WIKI_BASE = "https://en.wikipedia.org"
WIKI_SUMMARY_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"

HEADERS = {
    "User-Agent": "wiki-speedrunner/2.0-llm-semantic"
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


PAGE_CACHE = {}

def fetch_page(url: str) -> str:
    if url in PAGE_CACHE:
        return PAGE_CACHE[url]
    time.sleep(0.3)
    r = requests.get(url, timeout=20, headers=HEADERS)
    r.raise_for_status()
    PAGE_CACHE[url] = r.text
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

        # Filter out typical citation / metadata links
        if any(x in text.lower() for x in ["doi", "isbn", "pmid", "issn", "jstor", "hdl"]):
            continue

        # Ignore very short or non-alphabetic anchors
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
    except Exception:
        return title


# -------------------------------------------------------------------
# BASE PLAYER
# -------------------------------------------------------------------

@dataclass
class Player:
    """Base class with simple visualization printing."""

    def visualize_link(
        self,
        link: Dict[str, str],
        page_url: str,
        step: int,
        highlight_color: Optional[str] = None,
        delay: float = 0.0,
        is_candidate: bool = False,
    ):
        tags = []
        if is_candidate:
            tags.append("[CAND]")
        if highlight_color == "green":
            tags.append("[CHOSEN]")
        if link.get("non_mainspace"):
            tags.append("[NON-MAIN]")
        tag = " ".join(tags)
        print(f"  - {link['text']}  ({link['href']}) {tag}")
        if delay > 0:
            time.sleep(delay)


# -------------------------------------------------------------------
# SBERT HTTP CLIENT (OPTIONAL FOR SEMANTIC FILTERING)
# -------------------------------------------------------------------

class SBERTHTTPClient:
    """Simple HTTP client for the external SBERT server."""

    def __init__(self, base_url: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def encode(self, texts):
        single = False
        if isinstance(texts, str):
            texts = [texts]
            single = True

        payload = {
            "texts": texts,
            "normalize": True,
        }

        r = self.session.post(
            self.base_url + "/embed",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        arr = np.asarray(r.json()["embeddings"], dtype=np.float32)
        return arr[0] if single else arr


# -------------------------------------------------------------------
# LLM CLIENT
# -------------------------------------------------------------------

class LLMClient:
    """
    Simple LLM client that supports:
      - provider='local': OpenAI-compatible local server at base_url
      - provider='openrouter': OpenRouter API
    """

    def __init__(
        self,
        provider: str,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 6.0,
        max_retries: int = 4,
        min_call_interval: float = 0.0,  # seconds between calls to avoid hammering
    ):
        provider = provider.lower().strip()
        if provider not in {"local", "openrouter"}:
            raise ValueError("provider must be 'local' or 'openrouter'")

        self.provider = provider
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()
        self.last_raw_choice = None
        self.last_hidden_reasoning = None
        self.last_hidden_reasoning_details = None
        self.max_retries = max_retries
        self.min_call_interval = min_call_interval
        self._last_call_ts = 0.0

        if provider == "openrouter":
            self.base_url = base_url.rstrip("/") if base_url else "https://openrouter.ai/api"
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
            if not self.api_key:
                raise RuntimeError(
                    "OpenRouter provider selected but no API key found. "
                    "Set --llm-api-key or OPENROUTER_API_KEY env var."
                )
        else:
            # local OpenAI-compatible
            if not base_url:
                base_url = "http://localhost:11434"
            self.base_url = base_url.rstrip("/")
            self.api_key = api_key  # usually not needed for local

    def chat(self, messages, temperature: float = 0.2, max_tokens: int = 256) -> str:
        if self.provider == "openrouter":
            url = self.base_url + "/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        else:
            url = self.base_url + "/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # simple spacing between requests to be nice
        now = time.time()
        if self.min_call_interval > 0 and now - self._last_call_ts < self.min_call_interval:
            time.sleep(self.min_call_interval - (now - self._last_call_ts))

        last_error = None

        self.last_raw_choice = None
        self.last_hidden_reasoning = None
        self.last_hidden_reasoning_details = None

        for attempt in range(self.max_retries):
            try:
                r = self.session.post(url, json=payload, headers=headers, timeout=self.timeout)
                # Handle 429 explicitly without raising yet
                if r.status_code == 429:
                    # Try to respect Retry-After if present
                    retry_after = r.headers.get("Retry-After")
                    if retry_after is not None:
                        try:
                            delay = float(retry_after)
                        except ValueError:
                            delay = 2.0
                    else:
                        # exponential backoff: 1, 2, 4, 8...
                        delay = 2.0 ** attempt
                    time.sleep(delay)
                    last_error = f"429 Too Many Requests (attempt {attempt + 1}/{self.max_retries})"
                    continue

                # Any other non-2xx → raise
                r.raise_for_status()
                data = r.json()
                self._last_call_ts = time.time()

                # store raw response for debugging
                self.last_raw_choice = data

                # try to pull out any provider-side reasoning if it exists
                # (schema is provider-dependent, so keep this defensive)
                reasoning = data.get("reasoning")
                reasoning_details = data.get("reasoning_details")

                # some providers might nest this somewhere else; adapt as needed later
                self.last_hidden_reasoning = reasoning
                self.last_hidden_reasoning_details = reasoning_details

                content = data["choices"][0]["message"]["content"]
                return content if isinstance(content, str) else str(content)


            except Exception as e:
                last_error = str(e)
                # small backoff on random network issues
                time.sleep(1.0 * (attempt + 1))

        # If we get here, we failed all retries → let caller handle
        raise RuntimeError(f"LLM chat failed after {self.max_retries} retries. Last error: {last_error}")






# -------------------------------------------------------------------
# PLAYER 2 — SEMANTIC + LLM
# -------------------------------------------------------------------

from collections import Counter

class Player2SemanticLLM(Player):
    def __init__(
        self,
        llm_client: LLMClient,
        sbert_url: Optional[str] = None,
        shortlist_size: int = 8,
        summary_weight: float = 0.4,
        mcts_rollouts: int = 0,   # NEW: 0 = disabled
        mcts_depth: int = 3,      # NEW: lookahead depth
    ):
        super().__init__()
        self.llm = llm_client
        self.shortlist_size = shortlist_size
        self.summary_weight = summary_weight

        self.sbert = SBERTHTTPClient(sbert_url) if sbert_url else None
        self._target_emb_cache: Dict[str, np.ndarray] = {}
        self._summary_cache: Dict[str, Tuple[str, np.ndarray]] = {}

        self.run_log: List[Dict] = []
        self.llm_fail_count: int = 0

        # NEW: topic histogram to penalize overused themes
        self.topic_hist: Counter[str] = Counter()

        self.mcts_rollouts = mcts_rollouts
        self.mcts_depth = mcts_depth



    def __str__(self):
        if self.sbert:
            return "Player2_LLM+SBERT"
        return "Player2_LLM"

    # ---------- semantic helpers ----------

    def _candidate_tokens(self, cand: Dict[str, str]) -> Set[str]:
        title = title_from_url(cand.get("href", "") or "")
        summary = cand.get("summary") or ""
        anchor = cand.get("text") or cand.get("anchor") or ""
        text = " ".join([title, anchor, summary])
        return self._tokenize(text)

    def _hist_penalty(self, cand: Dict[str, str], penalty_scale: float = 0.05) -> float:
        """
        penalty_scale ~0.02–0.1 is usually reasonable.
        Higher = more aggressive avoidance of repeated topics.
        """
        toks = self._candidate_tokens(cand)
        if not toks:
            return 0.0
        # Sum frequencies of tokens we've seen before
        freq = sum(self.topic_hist[t] for t in toks)
        return penalty_scale * float(freq)

    def _augmented_score(self, cand: Dict[str, str], default_score: float = 1e9) -> float:
        """
        Base score (semantic distance) + histogram penalty.
        Lower is better.
        """
        base = float(cand.get("score", default_score))
        return base + self._hist_penalty(cand)
    
    def _update_topic_hist(self, chosen: Dict[str, str]):
        """
        Update histogram with tokens from the chosen candidate.
        """
        toks = self._candidate_tokens(chosen)
        self.topic_hist.update(toks)


    def _trim_summary(self, text: str, max_chars: int = 300) -> str:
        if not text:
            return ""

        text = text.strip()

        if len(text) <= max_chars:
            return text

        # Try keeping full sentences when possible
        sentences = re.split(r'(?<=[.!?]) +', text)
        out = ""

        for s in sentences:
            if len(out) + len(s) + 1 > max_chars:
                break
            out += s + " "

        out = out.strip()

        if len(out) < 50:
            # fallback to non-sentence trimming
            return text[:max_chars].rsplit(" ", 1)[0] + "…"

        return out + "…"

    def _embed_target(self, text: str) -> np.ndarray:
        key = text.strip()
        if key not in self._target_emb_cache:
            self._target_emb_cache[key] = self.sbert.encode(key)
        return self._target_emb_cache[key]

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return self.sbert.encode(texts)

    def _embed_summary(self, href: str) -> Tuple[str, np.ndarray]:
        key = normalize_wiki_url(href)
        if key in self._summary_cache:
            return self._summary_cache[key]
        summary = fetch_summary_for_target(key)
        emb = self.sbert.encode(summary)
        self._summary_cache[key] = (summary, emb)
        return summary, emb

    @staticmethod
    def _cosine_dist(M: np.ndarray, t: np.ndarray) -> np.ndarray:
        if M.size == 0:
            return np.zeros((0,), dtype=np.float32)
        return 1.0 - (M @ t)
    

    def _try_obvious_choice(
        self,
        candidates: List[Dict[str, str]],
        target_title: str,
        target_desc: str,
    ) -> Optional[int]:
        """
        Try to pick an 'obvious' candidate without calling the LLM.

        Heuristic:
          - Favor candidates whose title/summary contains the full target title.
          - Also count overlaps of long tokens (>= 6 chars) from target title/desc.
          - If a candidate has a strong score (>= 3), auto-select its index.
        """
        if not candidates:
            return None

        t_title = (target_title or "").lower()
        t_tokens = self._tokenize(target_title or "") | self._tokenize(target_desc or "")
        t_tokens = {tok for tok in t_tokens if len(tok) >= 6}

        best_idx = None
        best_score = 0

        for i, c in enumerate(candidates):
            title = title_from_url(c["href"]).lower()
            text = (title + " " + c.get("summary", "")).lower()

            score = 0
            # Full title match is very strong
            if t_title and t_title in text:
                score += 3

            # Bonus for overlapping long tokens
            for tok in t_tokens:
                if tok in text:
                    score += 1

            if score > best_score:
                best_score = score
                best_idx = i

        # Threshold can be tuned; 3 is "strongly related"
        if best_idx is not None and best_score >= 3:
            return best_idx

        return None

    def _log_step(
        self,
        step: int,
        page_url: str,
        target_url: str,
        candidates: List[Dict[str, str]],
        chosen: Dict[str, str],
        reasoning: str,
        source: str,
    ):
        """
        Append a structured JSON log entry for this step.
        source: 'auto_heuristic' | 'llm' | 'semantic_fallback' | 'direct_hit'
        """
        entry = {
            "step": step,
            "page": {
                "url": page_url,
                "title": title_from_url(page_url),
            },
            "target": {
                "url": target_url,
                "title": title_from_url(target_url),
            },
            "candidates": [],
            "selection": {
                "source": source,
                "reasoning": reasoning,
                "chosen": {
                    "href": chosen.get("href"),
                    "anchor": chosen.get("text"),
                    "title": title_from_url(chosen.get("href", "")),
                },
            },
        }

        for c in candidates:
            entry["candidates"].append({
                "href": c.get("href"),
                "anchor": c.get("text"),
                "title": title_from_url(c.get("href", "")),
                "summary": c.get("summary"),
                "score": c.get("score"),  # may be None if no SBERT
            })

        self.run_log.append(entry)

    # ---------- lexical fallback shortlist ----------

    @staticmethod
    def _tokenize(text: str) -> set:
        return set(re.findall(r"[a-zA-Z]{3,}", text.lower()))

    def _lexical_shortlist(
        self,
        links: List[Dict[str, str]],
        target_desc: str,
        max_size: int,
    ) -> List[Dict[str, str]]:
        if not links:
            return []

        target_tokens = self._tokenize(target_desc)
        if not target_tokens:
            return links[:max_size]

        scored = []
        for link in links:
            score = len(self._tokenize(link["text"]) & target_tokens)
            scored.append((score, link))

        scored.sort(key=lambda x: x[0], reverse=True)

        # If they’re all zero, just keep first N as-is
        if scored and scored[0][0] == 0:
            return links[:max_size]

        return [link for _, link in scored[:max_size]]

    # ---------- LLM selection ----------

    def _ask_llm_for_choice(
        self,
        current_title: str,
        target_title: str,
        target_desc: str,
        candidates: List[Dict[str, str]],
        show_reasoning: bool = False,
    ) -> Tuple[int, str]:
        """
        Returns: (choice_idx, reasoning_text)

        Enforces: the final choice index must be consistent with the reasoning.
        If the reasoning mentions a specific "Option N" that conflicts with the
        numeric CHOICE, we prefer the reasoning's option.
        """
        if not candidates:
            return 0, "(no candidates)"
    

        lines = []
        for i, link in enumerate(candidates, start=1):
            title = title_from_url(link["href"])
            raw_summary = link.get("summary", "").strip() or "(no summary available)"
            summary = self._trim_summary(raw_summary, max_chars=300)
            lines.append(
                f"{i}. ANCHOR: {link['text']}\n"
                f"   TITLE: {title}\n"
                f"   SUMMARY: {summary}"
            )

        candidates_str = "\n\n".join(lines)

        system_msg = {
            "role": "system",
            "content": (
                "You are playing the Wikipedia speedrun game.\n"
                "You must choose the single best next article to click to reach the target page in as few steps as possible.\n"
                "Use the titles and summaries to reason about which page moves closest in topic to the target.\n"
                "The option you argue is best in REASONING must match the final CHOICE number."
            ),
        }

        user_msg = {
            "role": "user",
            "content": (
                f"Current page: {current_title}\n"
                f"Target page: {target_title}\n"
                f"Target description:\n{target_desc}\n\n"
                f"Candidate next links:\n{candidates_str}\n\n"
                "Think step-by-step, then respond in the format:\n\n"
                "REASONING:\n"
                "<your reasoning, where you clearly state which OPTION NUMBER is best and why. "
                "The option number you argue is best MUST match the final CHOICE.>\n"
                "Your response must ALWAYS end with 'CHOICE: <number>' WITHOUT SINGLE QUOTES where <number> is the option number you choose.\n\n"
                "CHOICE: <number>\n"
            ),
        }

        try:
            reply = self.llm.chat(
                [system_msg, user_msg],
                temperature=0.0,
                max_tokens=2048,
            )
            # NEW: capture raw response object (store in LLMClient)
            raw = getattr(self.llm, "_last_raw_response", None)

            # Extract visible reasoning (from the prompt format)
            reasoning_match = re.search(r"REASONING:(.*?)(?:CHOICE:|\Z)", reply, re.S)
            reasoning_text = reasoning_match.group(1).strip() if reasoning_match else reply.strip()

            # Extract internal reasoning (if supported)
            internal_reasoning = None
            if raw:
                try:
                    choice = raw.get("choices", [{}])[0]
                    msg = choice.get("message") or {}

                    internal_reasoning = (
                        msg.get("reasoning")
                        or (msg.get("reasoning_details") or [{}])[0].get("text")
                    )
                    if internal_reasoning:
                        internal_reasoning = internal_reasoning.strip()
                except Exception:
                    pass
        except Exception as e:
            self.llm_fail_count += 1
            reasoning = f"(LLM error: {e}. Falling back to first candidate.)"
            return 0, reasoning

        # Extract reasoning text
        reasoning_match = re.search(r"REASONING:(.*?)(?:CHOICE:|\Z)", reply, re.S)
        reasoning_text = reasoning_match.group(1).strip() if reasoning_match else reply.strip()

        # Extract numeric CHOICE (if any)
        choice_match = re.search(r"CHOICE\s*:\s*(\d+)", reply)
        idx_from_choice = None
        if choice_match:
            try:
                idx_from_choice = int(choice_match.group(1)) - 1
            except ValueError:
                idx_from_choice = None

        # Extract any "Option N" mentions in reasoning
        mentioned_raw = re.findall(r"Option\s+(\d+)", reasoning_text, flags=re.I)
        mentioned_indices: List[int] = []
        for m in mentioned_raw:
            try:
                mi = int(m) - 1
                if 0 <= mi < len(candidates):
                    mentioned_indices.append(mi)
            except ValueError:
                continue

        idx: Optional[int] = None

        if mentioned_indices:
            # Most frequently mentioned index in reasoning
            counts = Counter(mentioned_indices)
            idx_from_reasoning = counts.most_common(1)[0][0]

            if idx_from_choice is not None and 0 <= idx_from_choice < len(candidates):
                # If CHOICE matches reasoning, cool; else trust reasoning
                if idx_from_choice == idx_from_reasoning:
                    idx = idx_from_choice
                else:
                    idx = idx_from_reasoning
            else:
                idx = idx_from_reasoning
        else:
            # No "Option N" text in reasoning – fall back to CHOICE number
            if idx_from_choice is not None and 0 <= idx_from_choice < len(candidates):
                idx = idx_from_choice

        # Final fallback if everything was scuffed
        if idx is None or not (0 <= idx < len(candidates)):
            idx = 0

        return idx, reasoning_text
    
    # ========== Monto Carlos Tree Search (simulated using SBERT because LLM quota) ===========

    def _simulate_greedy_walk(
        self,
        start_url: str,
        target_url: str,
        target_desc: str,
        depth: int,
        allow_non_mainspace: bool = False,
    ) -> Tuple[bool, int]:
        """
        Do a cheap, semantic-only greedy walk of up to `depth` steps.
        Returns (hit_target, steps_taken_if_hit_or_depth).
        No LLM calls, no logging, no topic_hist updates.
        """
        current = normalize_wiki_url(start_url)
        target_norm = normalize_wiki_url(target_url)
        visited = {current}

        for step in range(depth):
            if same_page(current, target_norm):
                return True, step

            try:
                html = fetch_page(current)
            except Exception:
                return False, depth

            links = extract_links(html, allow_non_mainspace=allow_non_mainspace)
            if not links:
                return False, depth

            # Minimal semantic shortlist: ignore LLM & hist, just SBERT if available.
            clean_links = []
            for l in links:
                href = normalize_wiki_url(l["href"])
                if href in visited:
                    continue
                l = dict(l)
                l["href"] = href
                clean_links.append(l)

            if not clean_links:
                return False, depth

            # If we see a direct hit, shortcut
            for l in clean_links:
                if same_page(l["href"], target_norm):
                    return True, step + 1

            # Simple lexical shortlist towards target_desc
            shortlist = self._lexical_shortlist(clean_links, target_desc, max_size=8)

            if self.sbert:
                try:
                    target_emb = self._embed_target(target_desc)
                    anchor_texts = [l["text"] for l in shortlist]
                    anchor_emb = self._embed_texts(anchor_texts)
                    anchor_dis = self._cosine_dist(anchor_emb, target_emb)
                    # choose best by distance
                    best_idx = int(np.argmin(anchor_dis))
                    next_link = shortlist[best_idx]
                except Exception:
                    next_link = shortlist[0]
            else:
                next_link = shortlist[0]

            current = next_link["href"]
            if current in visited:
                return False, depth
            visited.add(current)

        # Reached depth without hitting target
        return same_page(current, target_norm), depth
    
    def _score_candidate_with_rollouts(
        self,
        cand: Dict[str, str],
        target_url: str,
        target_desc: str,
        allow_non_mainspace: bool,
    ) -> float:
        """
        Lower is better. Combines:
          - success rate over rollouts
          - avg steps-to-hit (if hit)
          - base semantic score + histogram penalty
        """
        if self.mcts_rollouts <= 0:
            # just use existing augmented score
            return self._augmented_score(cand)

        start_url = cand["href"]
        hits = 0
        steps_sum = 0

        for _ in range(self.mcts_rollouts):
            try:
                hit, steps = self._simulate_greedy_walk(
                    start_url=start_url,
                    target_url=target_url,
                    target_desc=target_desc,
                    depth=self.mcts_depth,
                    allow_non_mainspace=allow_non_mainspace,
                )
            except Exception:
                continue

            if hit:
                hits += 1
                steps_sum += steps

        base = self._augmented_score(cand)

        if hits == 0:
            # Never hit in rollout: penalize candidate
            # big constant so actual hits are strongly preferred
            return base + 5.0
        else:
            avg_steps = steps_sum / hits
            # Reward early hits: base + avg_steps (lower is better)
            # You can tune the weight here if you want.
            return base + 0.5 * avg_steps

    # ---------- main decision function ----------

    def choose_next_link(
        self,
        target_url: str,
        target_desc: str,
        links: List[Dict[str, str]],
        visualize: bool,
        page_url: str,
        step: int,
        visited: Set[str],
        beam_width: int,
        **kwargs,
    ) -> Dict[str, str]:

        show_reasoning = kwargs.get("show_reasoning", False)

        target_norm = normalize_wiki_url(target_url)
        current_title = title_from_url(page_url)
        target_title = title_from_url(target_norm)

        if current_title == target_title:
            return None

        clean_links: List[Dict[str, str]] = []

        # 1. Filter visited & check direct hits
        for link in links:
            href = normalize_wiki_url(link["href"])
            if href in visited:
                continue

            if same_page(target_norm, href):
                if visualize:
                    self.visualize_link(link, page_url, step, "green", 0.2)
                # Log direct hit
                self._log_step(
                    step=step,
                    page_url=page_url,
                    target_url=target_url,
                    candidates=[link],
                    chosen=link,
                    reasoning="Direct link to target found; no LLM needed.",
                    source="direct_hit",
                )
                return link

            link["href"] = href
            clean_links.append(link)

        # If absolutely nothing left, just pick any unvisited or fallback to first
        if not clean_links:
            for link in links:
                if normalize_wiki_url(link["href"]) not in visited:
                    chosen = link
                    self._log_step(
                        step=step,
                        page_url=page_url,
                        target_url=target_url,
                        candidates=links,
                        chosen=chosen,
                        reasoning="No unvisited mainspace links remaining; choosing first unvisited.",
                        source="semantic_fallback",
                    )
                    self._update_topic_hist(chosen)
                    return chosen
            chosen = links[0]
            self._log_step(
                step=step,
                page_url=page_url,
                target_url=target_url,
                candidates=links,
                chosen=chosen,
                reasoning="Empty link set; falling back to first link.",
                source="semantic_fallback",
            )
            self._update_topic_hist(chosen)
            return chosen

        # 1b. DEDUPE by href
        dedup_links: List[Dict[str, str]] = []
        seen_hrefs = set()
        for l in clean_links:
            h = l["href"]
            if h in seen_hrefs:
                continue
            seen_hrefs.add(h)
            dedup_links.append(l)

        clean_links = dedup_links

        # 2. Decide how many candidates to show the LLM
        max_shortlist = min(len(clean_links), self.shortlist_size)

        candidates: List[Dict[str, str]] = []

        # 3. Semantic shortlist with SBERT (if available)
        if self.sbert:
            try:
                target_emb = self._embed_target(target_desc)
                anchor_texts = [l["text"] for l in clean_links]
                anchor_emb = self._embed_texts(anchor_texts)
                anchor_dis = self._cosine_dist(anchor_emb, target_emb)

                order = np.argsort(anchor_dis)
                # Respect beam_width if provided; otherwise default behavior
                if beam_width and beam_width > 0:
                    beam_k = min(len(order), beam_width)
                else:
                    beam_k = min(len(order), max_shortlist * 3)
                beam_idxs = list(order[:beam_k])

                # Collect summaries + cache embeddings
                summaries: List[str] = []
                summary_keys: List[str] = []
                missing_texts: List[str] = []
                missing_indices: List[int] = []

                for idx in beam_idxs:
                    link = clean_links[idx]
                    key = normalize_wiki_url(link["href"])
                    summary_keys.append(key)

                    if key in self._summary_cache:
                        summary, _ = self._summary_cache[key]
                        summaries.append(summary)
                    else:
                        summary = fetch_summary_for_target(key)
                        summaries.append(summary)
                        missing_texts.append(summary)
                        missing_indices.append(len(summaries) - 1)

                # Batch-embed only missing summaries
                if missing_texts:
                    missing_embs = self.sbert.encode(missing_texts)
                    if missing_embs.ndim == 1:
                        missing_embs = missing_embs.reshape(1, -1)

                    summary_embs = [None] * len(summaries)
                    m_i = 0
                    for i, key in enumerate(summary_keys):
                        if key in self._summary_cache:
                            _, emb = self._summary_cache[key]
                            summary_embs[i] = emb
                        else:
                            emb = missing_embs[m_i]
                            self._summary_cache[key] = (summaries[i], emb)
                            summary_embs[i] = emb
                            m_i += 1
                else:
                    summary_embs = [self._summary_cache[k][1] for k in summary_keys]

                cand_scores = []
                for local_idx, idx in enumerate(beam_idxs):
                    link = clean_links[idx]
                    sum_emb = summary_embs[local_idx]
                    sum_dis = self._cosine_dist(sum_emb.reshape(1, -1), target_emb)[0]
                    combined = (1 - self.summary_weight) * float(anchor_dis[idx]) + \
                               self.summary_weight * float(sum_dis)
                    cand_scores.append((combined, link, summaries[local_idx]))

                cand_scores.sort(key=lambda x: x[0])
                best = cand_scores[:max_shortlist]
                for combined, link, summary in best:
                    link = dict(link)
                    link["summary"] = summary
                    link["score"] = combined  # for logging + sanity checks
                    candidates.append(link)

            except Exception as e:
                print(f"[warn] SBERT semantic filtering failed: {e}")
                candidates = self._lexical_shortlist(clean_links, target_desc, max_shortlist)
        else:
            # No SBERT: lexical fallback
            candidates = self._lexical_shortlist(clean_links, target_desc, max_shortlist)

        # 4. Ensure every candidate has a summary (for LLM richness)
        for c in candidates:
            if "summary" not in c:
                c["summary"] = fetch_summary_for_target(c["href"])

        # 4b. Try obvious heuristic before calling LLM
        auto_idx = self._try_obvious_choice(candidates, target_title, target_desc)
        if auto_idx is not None:
            chosen = candidates[auto_idx]
            reasoning = (
                "Auto-selected candidate based on strong title/summary similarity "
                "to the target; no LLM call."
            )
            if visualize:
                print("  Candidate links considered (auto-heuristic):")
                for cand in candidates:
                    self.visualize_link(cand, page_url, step, is_candidate=True)
                print("\n  Auto-heuristic chose:")
                self.visualize_link(chosen, page_url, step, "green", 0.1)

            self._log_step(
                step=step,
                page_url=page_url,
                target_url=target_url,
                candidates=candidates,
                chosen=chosen,
                reasoning=reasoning,
                source="auto_heuristic",
            )
            self._update_topic_hist(chosen)
            return chosen

        if visualize:
            print("  Candidate links considered by LLM:")
            for cand in candidates:
                self.visualize_link(cand, page_url, step, is_candidate=True)

        # 5. Ask LLM to pick best candidate (if not disabled by repeated failures)
        if self.llm_fail_count >= 3:
            # Too many LLM errors: mcts fallback
            if self.mcts_rollouts > 0:
                chosen = min(
                    candidates,
                    key=lambda c: self._score_candidate_with_rollouts(
                        c,
                        target_url=target_url,
                        target_desc=target_desc,
                        allow_non_mainspace=False,
                    ),
                )
                reasoning = (
                    "LLM disabled due to repeated failures; using MCTS rollout refinement."
                )
            elif self.sbert and any("score" in c for c in candidates):
                chosen = min(candidates, key=lambda c: self._augmented_score(c))
                reasoning = (
                    "LLM disabled due to repeated failures; "
                    "using best semantic candidate (lowest score)."
                )

            else:
                chosen = candidates[0]
                reasoning = (
                    "LLM disabled due to repeated failures and no semantic score; "
                    "using first candidate."
                )

            if visualize:
                print("\n  Semantic fallback chose:")
                self.visualize_link(chosen, page_url, step, "green", 0.1)

            self._log_step(
                step=step,
                page_url=page_url,
                target_url=target_url,
                candidates=candidates,
                chosen=chosen,
                reasoning=reasoning,
                source="semantic_fallback",
            )
            self._update_topic_hist(chosen)
            return chosen

        choice_idx, reasoning = self._ask_llm_for_choice(
            current_title=current_title,
            target_title=target_title,
            target_desc=target_desc,
            candidates=candidates,
            show_reasoning=show_reasoning,
        )

        # First trust LLM
        chosen = candidates[choice_idx]

        # Then let MCTS refine it
        if self.mcts_rollouts > 0:
            chosen = min(
                candidates,
                key=lambda c: self._score_candidate_with_rollouts(
                    c,
                    target_url=target_url,
                    target_desc=target_desc,
                    allow_non_mainspace=False,
                ),
            )


        if show_reasoning:
            # 1) Parsed reasoning from the model's visible content
            print("\n===== LLM REASONING (parsed from content) =====")
            print(reasoning)
            print("==============================================\n")

            # 2) Actual provider-side reasoning, if available (OpenRouter, Qwen3, etc.)
            hidden = getattr(self.llm, "last_hidden_reasoning", None)
            hidden_details = getattr(self.llm, "last_hidden_reasoning_details", None)

            # Prefer `reasoning` if present
            if hidden and str(hidden).strip():
                print("===== LLM INTERNAL REASONING (provider `reasoning`) =====")
                print(hidden)
                print("=========================================================\n")
            elif isinstance(hidden_details, list) and hidden_details:
                # Many providers put the actual reasoning into reasoning_details[].text
                # We'll just join them for display purposes.
                texts = [
                    d.get("text", "")
                    for d in hidden_details
                    if isinstance(d, dict) and d.get("text")
                ]
                joined = "\n".join(texts).strip()
                if joined:
                    print("===== LLM INTERNAL REASONING (provider `reasoning_details`) =====")
                    print(joined)
                    print("=================================================================\n")


        if visualize:
            print("\n  LLM chose:")
            self.visualize_link(chosen, page_url, step, "green", 0.1)

        self._log_step(
            step=step,
            page_url=page_url,
            target_url=target_url,
            candidates=candidates,
            chosen=chosen,
            reasoning=reasoning,
            source="llm",
        )

        self._update_topic_hist(chosen)
        return chosen

    



# -------------------------------------------------------------------
# MAIN SPEEDRUN LOOP
# -------------------------------------------------------------------
def run_speedrun(
    start_url: str,
    target_url: str,
    target_desc: Optional[str],
    max_steps: int,
    visualize: bool,
    beam_width: int,
    allow_non_mainspace: bool,
    llm_provider: str,
    llm_model: str,
    llm_base_url: Optional[str],
    llm_api_key: Optional[str],
    sbert_url: Optional[str],
    shortlist_size: int,
    show_reasoning: bool,
):
    llm_client = LLMClient(
        provider=llm_provider,
        model=llm_model,
        base_url=llm_base_url,
        api_key=llm_api_key,
    )

    player = Player2SemanticLLM(
        llm_client=llm_client,
        sbert_url=sbert_url,
        shortlist_size=shortlist_size,
    )

    start_url = normalize_wiki_url(start_url)
    target_url = normalize_wiki_url(target_url)

    if not target_desc:
        print("[info] Fetching target description…")
        target_desc = fetch_summary_for_target(target_url)

    target_title = title_from_url(target_url)
    print(f"[player] {player}")
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
            show_reasoning=show_reasoning,
        )

        if next_link is None:
            print("Reached target!")
            break

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
    
    print("\n=== JSON RUN LOG ===")
    try:
        print(json.dumps(player.run_log, indent=2))
    except Exception as e:
        print(f"[warn] Failed to dump JSON run log: {e}")



# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Player2 semantic+LLM Wikipedia speedrunner")

    # Core game params
    ap.add_argument("--start", required=True, help="Start page (URL or title)")
    ap.add_argument("--target", required=True, help="Target page (URL or title)")
    ap.add_argument("--desc", default=None, help="Optional target description override")
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--no-viz", action="store_true")
    ap.add_argument("--beam-width", type=int, default=3)
    ap.add_argument("--allow-non-mainspace", action="store_true")

    # LLM config
    ap.add_argument(
        "--llm-provider",
        choices=["local", "openrouter"],
        default="local",
    )
    ap.add_argument(
        "--llm-model",
        required=True,
        help="Model name for the chosen LLM provider.",
    )
    ap.add_argument(
        "--llm-base-url",
        default=None,
        help="Base URL for LLM server. For local, something like 'http://localhost:11434'. "
             "For OpenRouter you can usually omit this.",
    )
    ap.add_argument(
        "--llm-api-key",
        default=None,
        help="API key for the LLM provider (needed for OpenRouter). "
             "If omitted for OpenRouter, OPENROUTER_API_KEY env var is used.",
    )

    # Semantic filter
    ap.add_argument(
        "--sbert-url",
        default=None,
        help="URL of SBERT embedding server for semantic pre-filtering (optional).",
    )
    ap.add_argument(
        "--shortlist-size",
        type=int,
        default=8,
        help="Number of candidates to show the LLM after semantic filtering.",
    )

    # Debug / reasoning
    ap.add_argument(
        "--show-reasoning",
        action="store_true",
        help="Print the LLM's chain-of-thought reasoning.",
    )

    args = ap.parse_args()

    run_speedrun(
        start_url=args.start,
        target_url=args.target,
        target_desc=args.desc,
        max_steps=args.max_steps,
        visualize=not args.no_viz,
        beam_width=args.beam_width,
        allow_non_mainspace=args.allow_non_mainspace,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        sbert_url=args.sbert_url,
        shortlist_size=args.shortlist_size,
        show_reasoning=args.show_reasoning,
    )


if __name__ == "__main__":
    main()
