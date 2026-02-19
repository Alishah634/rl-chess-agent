from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Optional
import requests

from tqdm import tqdm

API = "https://api.chess.com/pub"


@dataclass
class FetchCfg:
    target_games: int = 5000
    sleep_s: float = 1.0
    timeout_s: float = 30.0
    max_months_per_user: Optional[int] = None  # None = all months


def get_archives(session: requests.Session, username: str, cfg: FetchCfg) -> list[str]:
    url = f"{API}/player/{username}/games/archives"
    r = session.get(url, timeout=cfg.timeout_s)
    r.raise_for_status()
    return r.json().get("archives", [])


def download_month_pgn(session: requests.Session, month_archive_url: str, cfg: FetchCfg) -> str:
    # month_archive_url looks like: .../player/{u}/games/YYYY/MM
    pgn_url = month_archive_url.rstrip("/") + "/pgn"
    r = session.get(pgn_url, timeout=cfg.timeout_s)
    # be gentle on 429
    if r.status_code == 429:
        time.sleep(5)
        r = session.get(pgn_url, timeout=cfg.timeout_s)
    r.raise_for_status()
    return r.text


def count_games_in_pgn(pgn_text: str) -> int:
    # Simple heuristic: each game starts with [Event "..."]
    return pgn_text.count('\n[Event "') + (1 if pgn_text.startswith('[Event "') else 0)


def fetch_pgns(usernames: Iterable[str], out_path: str, cfg: FetchCfg) -> None:
    session = requests.Session()

    # REQUIRED for Chess.com API
    session.headers.update({
        "User-Agent": "ChessResearchBot/1.0 (your_email@example.com)"
    })
    total = 0

    with open(out_path, "w", encoding="utf-8") as out:
        for u in tqdm(usernames):
            archives = get_archives(session, u, cfg)
            # newest first helps you get modern games quickly
            archives = list(reversed(archives))

            if cfg.max_months_per_user is not None:
                archives = archives[: cfg.max_months_per_user]

            usernames_total = 0
            for month_url in archives:
                if total >= cfg.target_games:
                    print(f"Reached target: {total} games")
                    return

                pgn = download_month_pgn(session, month_url, cfg)
                n = count_games_in_pgn(pgn)

                if n == 0:
                    continue

                games = pgn.strip().split("\n\n[Event ")
                for i, g in enumerate(games):
                    if i > 0:
                        g = "[Event " + g  # add back removed header

                    # Keep only standard games
                    if '[Variant "' in g and '[Variant "Standard"]' not in g:
                        continue
                    out.write(g.strip() + "\n\n")
                    total += 1
                    usernames_total += 1

                # Evenly distribute the games amoung the GMs selected, note last GM likey to have the most games:
                if usernames_total >= cfg.target_games//len(usernames):
                    print(f"{usernames_total} games collected from {u}")
                    break 

                print(f"{u}: +{n} games from {month_url}  (total={total})")

                time.sleep(cfg.sleep_s)

    print(f"Done. Total games written: {total}")

if __name__ == "__main__":
    # Fill with Chess.com usernames (GMs, IMs, etc.)
    usernames = [
        "hikaru",
        "magnuscarlsen",
        "fabianocaruana",
        "anishgiri"
        # ...
    ]
    fetch_pgns(usernames, out_path="all_games.pgn", cfg=FetchCfg(target_games=5000, sleep_s=1.0))

