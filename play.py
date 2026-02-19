from __future__ import annotations

import sys
from typing import Optional #, Tuple

import chess
import chess.svg
import torch
from PyQt6.QtCore import Qt
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from policy import ChessPolicy, select_move  # uses your encode_board internally


def square_from_svg_click(widget: QSvgWidget, x: int, y: int, board_size_px: int = 520) -> Optional[chess.Square]:
    """
    Map a mouse click (x,y) within the SVG widget to a chess.Square.
    Assumes the SVG board is rendered with size ~= board_size_px and no rotation (white at bottom).
    """
    w = widget.width()
    h = widget.height()
    s = min(w, h)

    # center the board if widget is slightly larger
    x0 = (w - s) / 2.0
    y0 = (h - s) / 2.0

    fx = (x - x0) / s
    fy = (y - y0) / s
    if fx < 0 or fx >= 1 or fy < 0 or fy >= 1:
        return None

    file_ = int(fx * 8)          # 0..7 (a..h)
    rank_from_top = int(fy * 8)  # 0..7 (8..1)

    # python-chess: rank 0 is 1st rank, but our SVG has rank 8 at top
    rank_ = 7 - rank_from_top
    return chess.square(file_, rank_)


class PlayVsBotGUI(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Play vs RL Chess Bot")

        # --- Model setup ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessPolicy().to(self.device)
        self.model.eval()

        self.board = chess.Board()
        self.human_color = chess.WHITE

        self.selected_from: Optional[chess.Square] = None

        # --- UI ---
        self.svg = QSvgWidget()
        self.svg.setFixedSize(540, 540)
        self.svg.mousePressEvent = self.on_click  # type: ignore

        self.status = QLabel("Load a model, choose side, click squares to move.")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.side_box = QComboBox()
        self.side_box.addItems(["Play as White", "Play as Black"])
        self.side_box.currentIndexChanged.connect(self.on_side_change)

        self.btn_load = QPushButton("Load chess_policy.pt")
        self.btn_load.clicked.connect(self.load_model)

        self.btn_new = QPushButton("New Game")
        self.btn_new.clicked.connect(self.new_game)

        self.btn_bot_move = QPushButton("Bot Move Now")
        self.btn_bot_move.clicked.connect(self.bot_move_now)

        self.btn_undo = QPushButton("Undo (2 plies)")
        self.btn_undo.clicked.connect(self.undo_two_plies)

        top = QHBoxLayout()
        top.addWidget(self.btn_load)
        top.addWidget(self.side_box)
        top.addWidget(self.btn_new)
        top.addWidget(self.btn_undo)
        top.addWidget(self.btn_bot_move)

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addWidget(self.svg)
        layout.addWidget(self.status)
        self.setLayout(layout)

        self.render()

    def load_model(self) -> None:
        path = "chess_policy.pt"
        try:
            sd = torch.load(path, map_location=self.device)
            self.model.load_state_dict(sd)
            self.model.eval()
            self.status.setText(f"Loaded {path}.")
        except Exception as e:
            self.status.setText(f"Failed to load {path}: {e}")

        # If human chose black, bot should start
        self.maybe_bot_reply()

    def on_side_change(self) -> None:
        self.human_color = chess.WHITE if self.side_box.currentIndex() == 0 else chess.BLACK
        self.selected_from = None
        self.render()
        self.maybe_bot_reply()

    def new_game(self) -> None:
        self.board = chess.Board()
        self.selected_from = None
        self.render()
        self.maybe_bot_reply()

    def undo_two_plies(self) -> None:
        # undo bot + human if available
        if len(self.board.move_stack) >= 1:
            self.board.pop()
        if len(self.board.move_stack) >= 1:
            self.board.pop()
        self.selected_from = None
        self.render()
        self.status.setText("Undid last 2 plies (if available).")
        self.maybe_bot_reply()

    def bot_move_now(self) -> None:
        if self.board.is_game_over(claim_draw=True):
            self.status.setText(f"Game over: {self.board.outcome(claim_draw=True)}")
            return
        if self.board.turn == self.human_color:
            self.status.setText("It’s your turn.")
            return
        self.make_bot_move()

    def on_click(self, event) -> None:  # Qt passes a QMouseEvent
        if self.board.is_game_over(claim_draw=True):
            self.status.setText(f"Game over: {self.board.outcome(claim_draw=True)}")
            return

        # only allow human to move on their turn
        if self.board.turn != self.human_color:
            self.status.setText("Wait—bot is thinking/it’s bot’s turn.")
            return

        sq = square_from_svg_click(self.svg, event.position().x(), event.position().y())
        if sq is None:
            return

        # First click: select a piece
        if self.selected_from is None:
            piece = self.board.piece_at(sq)
            if piece is None:
                self.status.setText("Select a piece first.")
                return
            if piece.color != self.human_color:
                self.status.setText("That’s not your piece.")
                return
            self.selected_from = sq
            self.status.setText(f"Selected {chess.square_name(sq)}. Now click destination.")
            self.render()
            return

        # Second click: try to make a move
        from_sq = self.selected_from
        to_sq = sq
        self.selected_from = None

        move = self.resolve_move(from_sq, to_sq)
        if move is None:
            self.status.setText("Illegal move (or promotion not handled). Try again.")
            self.render()
            return

        self.board.push(move)
        self.status.setText(f"You played {move.uci()}")
        self.render()
        self.maybe_bot_reply()

    def resolve_move(self, from_sq: chess.Square, to_sq: chess.Square) -> Optional[chess.Move]:
        """
        Handle normal moves and basic promotion (defaults to queen).
        """
        # try non-promotion first
        candidate = chess.Move(from_sq, to_sq)
        if candidate in self.board.legal_moves:
            return candidate

        # promotion? (default to queen)
        piece = self.board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN:
            # If moving to last rank, try queen promo
            rank = chess.square_rank(to_sq)
            if (piece.color == chess.WHITE and rank == 7) or (piece.color == chess.BLACK and rank == 0):
                promo = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
                if promo in self.board.legal_moves:
                    return promo

        return None

    def maybe_bot_reply(self) -> None:
        # if it's bot's turn and game not over, make bot move
        if self.board.is_game_over(claim_draw=True):
            self.status.setText(f"Game over: {self.board.outcome(claim_draw=True)}")
            return
        if self.board.turn != self.human_color:
            self.make_bot_move()

    def make_bot_move(self) -> None:
        try:
            move, _ = select_move(self.model, self.board, self.device, greedy=True)
        except Exception as e:
            self.status.setText(f"Bot move failed: {e}")
            return

        self.board.push(move)
        self.status.setText(f"Bot played {move.uci()}")
        self.render()

    def render(self) -> None:
        last_move = self.board.peek() if self.board.move_stack else None
        svg_text = chess.svg.board(
            board=self.board,
            size=520,
            lastmove=last_move,
            squares=[self.selected_from] if self.selected_from is not None else None,
        )
        self.svg.load(bytearray(svg_text, encoding="utf-8"))


def main() -> None:
    app = QApplication(sys.argv)
    w = PlayVsBotGUI()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

