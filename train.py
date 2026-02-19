# train.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import os
import time
import torch
import chess
from tqdm import tqdm 

from policy import ChessPolicy, select_move

def terminal_result_white(board: chess.Board) -> int:
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return -0.5 
    return 1 if outcome.winner == chess.WHITE else -1


@dataclass
class StepLog:
    logp: torch.Tensor
    turn_was_white: bool

def demo_game_live(model, device, delay: float = 0.15, max_plies: int = 200) -> None:
    """
    Plays a single game using greedy moves (argmax) and prints the board each ply.
    """
    board = chess.Board()
    print("\n" + "=" * 60)
    print("DEMO GAME (greedy policy)")
    print(board, "\n")

    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        move, _ = select_move(model, board, device, greedy=True)  # greedy for readability
        board.push(move)
        plies += 1

        print(f"Ply {plies:3d} | {'White' if not board.turn else 'Black'} just played {move.uci()}")
        print(board, "\n")
        time.sleep(delay)

    print("DEMO RESULT:", board.outcome(claim_draw=True))
    print("=" * 60 + "\n")

def play_one_game(model: ChessPolicy, device: torch.device, max_plies: int = 400) -> Tuple[List[StepLog], int]:
    board = chess.Board()
    traj: List[StepLog] = []

    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        turn_was_white = (board.turn == chess.WHITE)
        move, logp = select_move(model, board, device, greedy=False)
        traj.append(StepLog(logp=logp, turn_was_white=turn_was_white))
        board.push(move)
        plies += 1

    # If we hit max plies, treat as draw (helps avoid endless games early on)
    if not board.is_game_over(claim_draw=True):
        # z = 0
        z = -0.5
    else:
        z = terminal_result_white(board)

    return traj, z


def train(games: int = 500, lr: float = 1e-3, device_str: str = "cuda") -> ChessPolicy:
    # device = torch.device(device_str)
    
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model = ChessPolicy().to(device)

    if os.path.exists("chess_policy_pretrained.pt"):
        # model.load_state_dict(torch.load("chess_policy_pretrained.pt", map_location=device))
        model.load_state_dict(torch.load("checkpoints/checkpoint_500.pt", map_location=device))
        print("Loaded pretrained model: chess_policy_pretrained.pt")

    print("Model device:", next(model.parameters()).device)  # should be cuda:0
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for g in tqdm(range(games), miniters=25, mininterval=0):
        traj, z = play_one_game(model, device)

        # REINFORCE: -sum(G_t * logp_t), where G_t is result from the mover's perspective 
        loss = torch.zeros((), device=device)
        for step in traj:
            G = z if step.turn_was_white else -z
            loss = loss + (-float(G) * step.logp)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        if (g + 1) % 50 == 0:
            demo_game_live(model, device, delay=0.12, max_plies=160)
            torch.save(model.state_dict(), f"checkpoints/checkpoint_{g+1}.pt")
            print(f"game {g+1}/{games}  result(z)={z}  plies={len(traj)}  loss={loss.item():.3f}")

    torch.save(model.state_dict(), "chess_policy.pt")
    print("Saved: chess_policy.pt")
    return model

if __name__ == "__main__":

    # Path to store checkpoints of the model:
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
        print("Created checkpoint directory!")

    train(games=5000, lr=1e-3, device_str="cuda")

