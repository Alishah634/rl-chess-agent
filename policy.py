# policy.py
from __future__ import annotations
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess

from encoding import encode_board


def move_features(board: chess.Board, move: chess.Move) -> np.ndarray:
    promo_map = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}

    feats = np.array(
        [
            move.from_square / 63.0,
            move.to_square / 63.0,
            promo_map.get(move.promotion, 0) / 4.0,
            1.0 if board.is_capture(move) else 0.0,
            1.0 if board.is_castling(move) else 0.0,
            1.0 if board.is_en_passant(move) else 0.0,
            0.0,  # gives check (computed below)
        ],
        dtype=np.float32,
    )

    board.push(move)
    feats[6] = 1.0 if board.is_check() else 0.0
    board.pop()
    return feats


class ChessPolicy(nn.Module):
    def __init__(self, move_feat_dim: int = 7, hidden: int = 128) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(18, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.board_fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, hidden),
            nn.ReLU(),
        )
        self.move_scorer = nn.Sequential(
            nn.Linear(hidden + move_feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, board_tensor: torch.Tensor, move_feats: torch.Tensor) -> torch.Tensor:
        # board_tensor: (1,18,8,8), move_feats: (N,7) -> logits: (N,)
        h = self.board_fc(self.conv(board_tensor))   # (1,H)
        h = h.repeat(move_feats.size(0), 1)          # (N,H)
        x = torch.cat([h, move_feats], dim=1)        # (N,H+7)
        return self.move_scorer(x).squeeze(1)        # (N,)


def select_move(
    model: ChessPolicy,
    board: chess.Board,
    device: torch.device,
    greedy: bool = False,
) -> Tuple[chess.Move, torch.Tensor]:
    legal_moves: List[chess.Move] = list(board.legal_moves)
    if not legal_moves:
        raise RuntimeError("No legal moves")

    s_np = encode_board(board)                       # (18,8,8)
    s = torch.from_numpy(s_np).unsqueeze(0).to(device)  # (1,18,8,8)

    m_np = np.stack([move_features(board, mv) for mv in legal_moves], axis=0)  # (N,7)
    m = torch.from_numpy(m_np).to(device)

    logits = model(s, m)               # (N,)
    probs = F.softmax(logits, dim=0)   # (N,)

    if greedy:
        idx = torch.argmax(probs)
        logp = torch.log(probs[idx] + 1e-12)
        return legal_moves[int(idx.item())], logp

    dist = torch.distributions.Categorical(probs=probs)
    idx = dist.sample()
    logp = dist.log_prob(idx)
    return legal_moves[int(idx.item())], logp

