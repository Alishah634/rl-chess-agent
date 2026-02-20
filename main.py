# train.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import os
import time
import torch
import torch.nn.functional as F
import chess
from tqdm import tqdm 

from policy import ChessPolicy, select_move, get_material_score

def terminal_result_white(board: chess.Board) -> float:
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None: # If Draw!
        # -0.5 teaches worse than loss, better then win, 0.0 is completely neutral, -0.8 teach its almost as bad as losing(risky and aggressive)
        return -0.5
    return 1 if outcome.winner == chess.WHITE else -1


@dataclass
class StepLog:
    logp: torch.Tensor
    value: torch.Tensor # Store the Critic's guess
    entropy: torch.Tensor  # enciourages exploration by maximizing this values
    turn_was_white: bool
    step_reward: float # Reward for the move played based on the matrial advantage/disadvantage due to the last move played

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
        move, logp, value, entropy = select_move(model, board, device, greedy=True)  # greedy for readability
        board.push(move)
        plies += 1

        print(f"Ply {plies:3d} | {'White' if not board.turn else 'Black'} just played {move.uci()}")
        print(board, "\n")
        time.sleep(delay)

    print("DEMO RESULT:", board.outcome(claim_draw=True))
    print("=" * 60 + "\n")

def play_one_game(model: ChessPolicy, device: torch.device, max_plies: int = 400) -> Tuple[List[StepLog], float]:
    board = chess.Board()
    traj: List[StepLog] = []

    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        turn_was_white = (board.turn == chess.WHITE)
        move, logp, value, entropy = select_move(model, board, device, greedy=False)

        # Try evaluating the matrial value that a player has on the board:
        material_before = get_material_score(board)
        board.push(move)
        material_after = get_material_score(board)

        # If I am White, I want the score to go UP. If I am Black, I want it to go DOWN.
        delta = material_after - material_before
        step_reward = delta if turn_was_white else -delta

        traj.append(StepLog(logp=logp, value=value, entropy=entropy, turn_was_white=turn_was_white, step_reward=step_reward))

        plies += 1

    z = terminal_result_white(board)

    return traj, z


def train(games: int = 500, lr: float = 1e-3, device_str: str = "cuda") -> ChessPolicy:
    # device = torch.device(device_str)
    
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model = ChessPolicy().to(device)

    if os.path.exists("chess_policy_pretrained.pt"):
        # model.load_state_dict(torch.load("chess_policy_pretrained.pt", map_location=device))
        # model.load_state_dict(torch.load("checkpoints/checkpoint_500.pt", map_location=device))
        print("Loaded pretrained model: chess_policy_pretrained.pt")

    print("Model device:", next(model.parameters()).device)  # should be cuda:0
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Hold stats for who has one or lost a game: 
    stats = {"white_wins": 0, "black_wins": 0, "draws": 0, "total_plies": 0}
    window_size = 50
    for g in tqdm(range(games), miniters=25, mininterval=0):
        traj, z = play_one_game(model, device)
        
        policy_loss = list()
        value_loss = list()

        # Update the stats for how many games were won or lost:
        if z == 1:
            stats["white_wins"] += 1
        elif z == -1:
            stats["black_wins"] += 1
        else:
            stats["draws"] += 1

        stats["total_plies"] += len(traj)

        for step in traj:
            # G the actual outcome from perspective of the player whose turn it was:
            G = torch.tensor([z if step.turn_was_white else -z], device=device).float()

            # "Dopamine hit" from the move itself G = (Final Outcome) + (Material gained/lost in this move)
            G += step.step_reward

            # Advantage = outcome - predicted outcome:
            # This is a small penalty for long games so itll try to win in the fewwest moves it can:
            time_penalty = len(traj) * 0.001
            advantage = (G - time_penalty) - step.value.detach()
        
            # Whats teh actor loss?:
            p_loss = -step.logp * advantage

            # Whats the critic loss?:
            v_loss = F.mse_loss(step.value.squeeze(), G.squeeze())

            # entropy, subtract as we want to maximize the entopy so it keeps exploring states until convergence:
            entropy_loss = -0.05* step.entropy 

            policy_loss.append(p_loss + entropy_loss)
            value_loss.append(v_loss)

        loss = torch.stack(policy_loss).mean() + 0.5 * torch.stack(value_loss).mean()
    
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        if (g + 1) % window_size == 0:
            demo_game_live(model, device, delay=0.12, max_plies=160)
            torch.save(model.state_dict(), f"checkpoints/checkpoint_{g+1}.pt")
            print(f"game {g+1}/{games}  result(z)={z}  plies={len(traj)}  loss={loss.item():.3f}")
            
            avg_plies = stats["total_plies"] / window_size
            total_wins = stats["white_wins"] + stats["black_wins"]
            status = "Shuffling"
            if total_wins == 0 and avg_plies < 30:
                status = "SUicide/Penalty Avoidance"
            elif total_wins > 5 and 40 <= avg_plies <= 150:
                status = "GOLDEN ZONE, JUST RIGHT WHERE IT SHOULD BE!"
            elif avg_plies >= 380:
                status = "Maximum Ply Stalling"

            log_msg = (f"Games {g-48}-{g+1} | Wins: {total_wins} | Draws: {stats['draws']} | "
                       f"Avg Plies: {avg_plies:.1f} | Status: {status}\n")

            with open("win_stats.log", mode='a') as f:
                f.write(log_msg)
            # Reset stats for the next window
            stats = {"white_wins": 0, "black_wins": 0, "draws": 0, "total_plies": 0}


    torch.save(model.state_dict(), "chess_policy.pt")
    print("Saved: chess_policy.pt")
    return model

if __name__ == "__main__":

    # Path to store checkpoints of the model:
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
        print("Created checkpoint directory!")

    train(games=5000, lr=1e-3, device_str="cuda")

# import chess
# from encoding import encode_board
#
#
# class RLAgent():
#     def __init__(self, board) -> None:
#         # super().__init__()
#         # self.action 
#         # Moves played so far i.e game_history
#         self.board = board
#         self.game_history = list()
#         self.action = self.get_action()
#
#         print(self.action)
#
#     def get_action(self):
#         return list(self.board.legal_moves)
#         # return action
#
#
#
# if __name__ == "__main__":
#
#     # Intialize board:
#     board = chess.Board()
#     x = encode_board(board)
#     print(x.)
#
#
#
#
#
#     # board.push_san("Qh5")
#
