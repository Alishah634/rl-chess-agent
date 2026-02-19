from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import chess
import chess.pgn
import numpy as np
import torch
import torch.nn.functional as F

from encoding import encode_board
from policy import ChessPolicy, move_features


def iter_games(pgn_path: Path, max_games: Optional[int]) -> Tuple[int, chess.pgn.Game]:
    with pgn_path.open("r", encoding="utf-8", errors="ignore") as f:
        i = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            i += 1
            if max_games is not None and i > max_games:
                break
            yield i, game


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True, help="Path to PGN file containing many games")
    ap.add_argument("--out", default="chess_policy_pretrained.pt", help="Output weights file")
    ap.add_argument("--max-games", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--print-every", type=int, default=200)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    model = ChessPolicy().to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    pgn_path = Path(args.pgn)
    if not pgn_path.exists():
        raise FileNotFoundError(pgn_path)

    step = 0
    used_positions = 0
    skipped_positions = 0

    for ep in range(args.epochs):
        print(f"\nEpoch {ep+1}/{args.epochs}")

        for game_idx, game in iter_games(pgn_path, args.max_games):
            board = game.board()

            for mv in game.mainline_moves():
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    break

                # find the PGN move in the legal move list
                try:
                    target_idx = legal_moves.index(mv)
                except ValueError:
                    skipped_positions += 1
                    break  # game probably corrupted/variant/pgn mismatch

                # state tensor
                s_np = encode_board(board)  # (18,8,8) float32
                s = torch.from_numpy(s_np).unsqueeze(0).to(device)  # (1,18,8,8)

                # legal move features (Option A)
                m_np = np.stack([move_features(board, lm) for lm in legal_moves], axis=0)  # (N,7)
                m = torch.from_numpy(m_np).to(device)

                logits = model(s, m)  # (N,)
                # cross-entropy expects (batch, classes)
                loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([target_idx], device=device))

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                used_positions += 1
                step += 1

                board.push(mv)

            if game_idx % args.print_every == 0:
                print(
                    f"games={game_idx} steps={step} used_positions={used_positions} "
                    f"skipped_positions={skipped_positions}"
                )

    torch.save(model.state_dict(), args.out)
    print(f"\nSaved pretrained weights to: {args.out}")
    print(f"Used positions: {used_positions} | Skipped positions: {skipped_positions}")


if __name__ == "__main__":
    # example usage:
    # python pretrain_pgn.py --pgn my_games.pgn --max-games 5000 --epochs 2 --lr 1e-3

    main()

