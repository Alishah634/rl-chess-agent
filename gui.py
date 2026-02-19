from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional

import chess
import chess.svg
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtSvgWidgets import QSvgWidget


@dataclass
class DemoConfig:
    delay_ms: int = 150
    max_plies: int = 160


class DemoWorker(QThread):
    # emits: svg_text, status_text
    frame = pyqtSignal(str, str)
    finished_status = pyqtSignal(str)

    def __init__(
        self,
        model_provider,  # callable -> model
        select_move_fn,  # (model, board, device, greedy=True) -> (move, logp)
        device,
        cfg: DemoConfig,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._model_provider = model_provider
        self._select_move = select_move_fn
        self._device = device
        self._cfg = cfg
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        self._stop = False
        model = self._model_provider()

        board = chess.Board()
        plies = 0

        # initial frame
        self._emit_board(board, plies, "Demo started")

        while not board.is_game_over(claim_draw=True) and plies < self._cfg.max_plies and not self._stop:
            move, _ = self._select_move(model, board, self._device, greedy=True)
            board.push(move)
            plies += 1

            side = "White" if board.turn == chess.WHITE else "Black"
            status = f"Ply {plies}/{self._cfg.max_plies} | Next to move: {side} | Last: {move.uci()}"
            self._emit_board(board, plies, status)

            self.msleep(self._cfg.delay_ms)

        outcome = board.outcome(claim_draw=True)
        self.finished_status.emit(f"Demo finished: {outcome}")

    def _emit_board(self, board: chess.Board, plies: int, status: str) -> None:
        svg = chess.svg.board(board=board, size=520)
        self.frame.emit(svg, status)


class ChessDemoGUI(QWidget):
    def __init__(self, model_provider, select_move_fn, device) -> None:
        super().__init__()
        self.setWindowTitle("RL Chess Bot — Live Demo (every N games)")

        self.svg_widget = QSvgWidget()
        self.svg_widget.setFixedSize(540, 540)

        self.status = QLabel("Idle")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.interval_box = QSpinBox()
        self.interval_box.setRange(1, 10000)
        self.interval_box.setValue(50)

        self.delay_box = QSpinBox()
        self.delay_box.setRange(10, 2000)
        self.delay_box.setValue(150)

        self.maxplies_box = QSpinBox()
        self.maxplies_box.setRange(20, 2000)
        self.maxplies_box.setValue(160)

        self.start_btn = QPushButton("Run Demo Now")
        self.stop_btn = QPushButton("Stop Demo")
        self.stop_btn.setEnabled(False)

        layout = QVBoxLayout()
        layout.addWidget(self.svg_widget)
        layout.addWidget(self.status)

        layout.addWidget(QLabel("Demo interval (games):"))
        layout.addWidget(self.interval_box)
        layout.addWidget(QLabel("Move delay (ms):"))
        layout.addWidget(self.delay_box)
        layout.addWidget(QLabel("Max plies:"))
        layout.addWidget(self.maxplies_box)

        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        self.setLayout(layout)

        self._model_provider = model_provider
        self._select_move_fn = select_move_fn
        self._device = device

        self._worker: Optional[DemoWorker] = None

        self.start_btn.clicked.connect(self.run_demo)
        self.stop_btn.clicked.connect(self.stop_demo)

    def run_demo(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return

        cfg = DemoConfig(
            delay_ms=int(self.delay_box.value()),
            max_plies=int(self.maxplies_box.value()),
        )
        self._worker = DemoWorker(self._model_provider, self._select_move_fn, self._device, cfg)
        self._worker.frame.connect(self._on_frame)
        self._worker.finished_status.connect(self._on_finished)
        self._worker.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status.setText("Starting demo...")

    def stop_demo(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
        self.status.setText("Stopping demo...")

    def _on_frame(self, svg_text: str, status: str) -> None:
        self.svg_widget.load(bytearray(svg_text, encoding="utf-8"))
        self.status.setText(status)

    def _on_finished(self, msg: str) -> None:
        self.status.setText(msg)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def demo_interval(self) -> int:
        return int(self.interval_box.value())


def run_gui(model_provider, select_move_fn, device) -> None:
    app = QApplication(sys.argv)
    w = ChessDemoGUI(model_provider, select_move_fn, device)
    w.show()
    sys.exit(app.exec())

