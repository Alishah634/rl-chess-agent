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
    def __init__(self, move_feat_dim: int = 7, hidden: int = 256) -> None:
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(18, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # Increased to 128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
        )

        # 1. DUMMY PASS to find the exact flattened size
        with torch.no_grad():
            dummy_board = torch.zeros(1, 18, 8, 8)
            self.flatten_dim = self.conv(dummy_board).shape[1] 
            # This will correctly be 128 * 8 * 8 = 8192

        # 2. Heads using the calculated dimension
        self.policy_fc = nn.Linear(self.flatten_dim + move_feat_dim, hidden)
        self.policy_out = nn.Linear(hidden, 1)
        
        self.value_fc = nn.Linear(self.flatten_dim, hidden)
        self.value_out = nn.Linear(hidden, 1)

    def forward(self, board_tensor, move_feats):
        board_features = self.conv(board_tensor) # (1, 8192)
        
        # Critic (Value Head)
        v_h = F.relu(self.value_fc(board_features))
        value = torch.tanh(self.value_out(v_h)) 
        
        # Actor (Policy Head)
        h_policy = board_features.repeat(move_feats.size(0), 1) # (N, 8192)
        x = torch.cat([h_policy, move_feats], dim=1) # (N, 8199)
        logits = self.policy_out(F.relu(self.policy_fc(x))).squeeze(1)
        
        return logits, value

def select_move(
    model: ChessPolicy,
    board: chess.Board,
    device: torch.device,
    greedy: bool = False,
) -> Tuple[chess.Move, torch.Tensor, torch.Tensor, torch.Tensor]: # Added entropy
    legal_moves = list(board.legal_moves)
    
    s_np = encode_board(board)
    s = torch.from_numpy(s_np).unsqueeze(0).to(device).float()
    m_np = np.stack([move_features(board, mv) for mv in legal_moves], axis=0)
    m = torch.from_numpy(m_np).to(device)
        

    temp = 1.5 # Higher = more random/exploratory, I really should make this a parameter....
    logits, value = model(s, m) 
    probs = F.softmax(logits / temp, dim=0)
    dist = torch.distributions.Categorical(probs=probs)

    if greedy:
        idx = torch.argmax(probs)
    else:
        idx = dist.sample()
    
    logp = dist.log_prob(idx)
    entropy = dist.entropy() # This measures how "uncertain" the AI is
    
    return legal_moves[int(idx.item())], logp, value, entropy

def get_material_score(board: chess.Board) -> int:
    # Standard values
    values = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0} # King is 0 or ignored
    score = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            val = values.get(piece.piece_type, 0)
            score += val if piece.color == chess.WHITE else -val
    return score
