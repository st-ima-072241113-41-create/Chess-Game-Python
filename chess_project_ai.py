"""
Chess Game: Human vs AI
Language: Python
GUI: Tkinter
Rules library: python-chess

Numeric piece codes:
  empty = 0.0
  pawn 1.0 | knight 3.0 | bishop 3.25 | rook 5.0 | queen 9.0 | king 10.0
  White pieces are positive; Black pieces are negative.

Install:
    python -m pip install chess

Run:
    python chess_project_ai.py
    python chess_project_ai.py --test
"""
import argparse
import math
import random
import sys
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox

try:
    import chess
except ImportError:
    sys.exit(
        "This program needs python-chess.\n"
        "Install it with: python -m pip install chess"
    )

PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.25,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 10.0,
}
PIECE_TO_CODE = {}
for _ptype, _value in PIECE_VALUES.items():
    PIECE_TO_CODE[(_ptype, chess.WHITE)] = _value
    PIECE_TO_CODE[(_ptype, chess.BLACK)] = -_value

LIGHT_SQUARE = "#F0D9B5"
DARK_SQUARE = "#B58863"
SELECTED_SQUARE = "#F7EC6E"
LEGAL_SQUARE = "#A9D18E"
CAPTURE_SQUARE = "#E58B8B"
SHADOW_COLOR = "#222222"
NUMBER_COLOR = "#202020"
SYMBOL_FONT_FAMILY = "Segoe UI Symbol"
PIECE_SYMBOLS = {
    chess.PAWN: "\u265F",
    chess.KNIGHT: "\u265E",
    chess.BISHOP: "\u265D",
    chess.ROOK: "\u265C",
    chess.QUEEN: "\u265B",
    chess.KING: "\u265A",
}


def board_to_numeric_matrix(board):
    """Return an 8x8 matrix, starting at rank 8."""
    matrix = []
    for row in range(8):
        rank = 7 - row
        line = []
        for col in range(8):
            piece = board.piece_at(chess.square(col, rank))
            line.append(
                0.0 if piece is None
                else PIECE_TO_CODE[(piece.piece_type, piece.color)]
            )
        matrix.append(line)
    return matrix


def evaluate_board(board):
    """Positive score favours White; negative score favours Black."""
    if board.is_checkmate():
        return -100000 if board.turn == chess.WHITE else 100000
    if board.is_game_over(claim_draw=True):
        return 0

    score = 0.0
    for piece_type, value in PIECE_VALUES.items():
        score += value * len(board.pieces(piece_type, chess.WHITE))
        score -= value * len(board.pieces(piece_type, chess.BLACK))

    # Small bonuses for mobility and controlling the centre.
    score += 0.03 * (board.legal_moves.count() if board.turn == chess.WHITE else -board.legal_moves.count())
    centre = (chess.D4, chess.E4, chess.D5, chess.E5)
    for sq in centre:
        piece = board.piece_at(sq)
        if piece:
            score += 0.15 if piece.color == chess.WHITE else -0.15
    return score


def ordered_moves(board):
    """Try captures and promotions first to improve alpha-beta search."""
    moves = list(board.legal_moves)

    def priority(move):
        score = 0
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            if victim:
                score += 10 * PIECE_VALUES[victim.piece_type]
            if attacker:
                score -= PIECE_VALUES[attacker.piece_type]
        if move.promotion:
            score += 20
        if board.gives_check(move):
            score += 1
        return score

    moves.sort(key=priority, reverse=True)
    return moves


def minimax(board, depth, alpha, beta):
    """Alpha-beta minimax. White maximizes; Black minimizes."""
    if depth == 0 or board.is_game_over(claim_draw=True):
        return evaluate_board(board), None

    best_move = None
    moves = ordered_moves(board)

    if board.turn == chess.WHITE:
        best_score = -math.inf
        for move in moves:
            board.push(move)
            score, _ = minimax(board, depth - 1, alpha, beta)
            board.pop()
            if score > best_score:7
                best_score, best_move = score, move
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break
    else:
        best_score = math.inf
        for move in moves:
            board.push(move)
            score, _ = minimax(board, depth - 1, alpha, beta)
            board.pop()
            if score < best_score:
                best_score, best_move = score, move
            beta = min(beta, best_score)
            if beta <= alpha:
                break

    return best_score, best_move


def choose_ai_move(board, depth=2):
    """Choose a move for Black using minimax and alpha-beta pruning."""
    if board.is_game_over(claim_draw=True):
        return None
    _, move = minimax(board, depth, -math.inf, math.inf)
    return move


class ChessGame:
    def __init__(self, root, square_size=None, ai_depth=2, auto_queen=False):
        self.root = root
        self.ai_depth = max(1, min(ai_depth, 3))
        self.auto_queen = auto_queen
        self.root.title("Chess Project - Human vs AI")
        self.root.minsize(440, 560)

        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        if square_size is None:
            available = min(screen_w * 0.9, screen_h * 0.75 - 170)
            square_size = int(max(40, min(110, available // 8)))

        self.board = chess.Board()
        self.selected_square = None
        self.legal_targets = set()
        self.flipped = False
        self.matrix = []
        self.moves_from = {}
        self.outcome = None
        self.result_shown = False
        self.base_status = ""
        self.matrix_window = None
        self.matrix_text = None
        self.sq_size = square_size
        self.ox = self.oy = 0
        self.drawn = {}
        self._resize_job = None
        self.ai_thinking = False

        tk.Label(
            root, text="CHESS: YOU (WHITE) vs AI (BLACK)",
            font=("Arial", 16, "bold"), bg="#243447", fg="white", pady=10
        ).pack(fill="x")
        self.status_label = tk.Label(
            root, text="", font=("Arial", 11, "bold"), pady=8, padx=8
        )
        self.status_label.pack(fill="x")
        self.status_label.bind(
            "<Configure>",
            lambda e: self.status_label.config(wraplength=max(e.width - 16, 100)),
        )

        self.piece_font = tkfont.Font(
            root=root, family=SYMBOL_FONT_FAMILY, size=-max(int(square_size * 0.62), 12)
        )
        self.number_font = tkfont.Font(
            root=root, family="Arial", size=-max(int(square_size * 0.20), 8), weight="bold"
        )
        self.canvas = tk.Canvas(
            root, width=square_size * 8, height=square_size * 8,
            highlightthickness=0, bg=root.cget("bg")
        )
        self.canvas.pack(fill="both", expand=True, padx=12, pady=4)

        self.rect_ids, self.text_ids, self.shadow_ids, self.num_ids = {}, {}, {}, {}
        for sq in chess.SQUARES:
            self.rect_ids[sq] = self.canvas.create_rectangle(0, 0, 0, 0, width=0)
            self.shadow_ids[sq] = self.canvas.create_text(
                0, 0, font=self.piece_font, fill=SHADOW_COLOR
            )
            self.text_ids[sq] = self.canvas.create_text(
                0, 0, font=self.piece_font, fill="#000000"
            )
            self.num_ids[sq] = self.canvas.create_text(
                0, 0, font=self.number_font, fill=NUMBER_COLOR
            )

        controls = tk.Frame(root, pady=10)
        controls.pack()
        for label, command in (
            ("New Game", self.new_game),
            ("Undo", self.undo),
            ("Flip Board", self.flip_board),
            ("Numeric Matrix", self.show_matrix),
        ):
            tk.Button(
                controls, text=label, font=("Arial", 10),
                command=command, padx=8
            ).pack(side="left", padx=3)

        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-n>", lambda e: self.new_game())
        self.root.bind("<f>", lambda e: self.flip_board())
        self.after_position_change()

    def square_to_cell(self, square):
        file_, rank = chess.square_file(square), chess.square_rank(square)
        return (rank, 7 - file_) if self.flipped else (7 - rank, file_)

    def cell_to_square(self, row, col):
        return chess.square(7 - col, row) if self.flipped else chess.square(col, 7 - row)

    def code_at(self, square):
        return self.matrix[7 - chess.square_rank(square)][chess.square_file(square)]

    def on_canvas_resize(self, _event):
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(30, self.layout_board)

    def layout_board(self):
        self._resize_job = None
        width, height = max(self.canvas.winfo_width(), 1), max(self.canvas.winfo_height(), 1)
        size = max(min(width, height) // 8, 8)
        self.sq_size = size
        self.ox, self.oy = (width - size * 8) // 2, (height - size * 8) // 2
        self.piece_font.configure(size=-max(int(size * 0.62), 12))
        self.number_font.configure(size=-max(int(size * 0.20), 8))
        shadow = max(1, size // 40)
        for sq in chess.SQUARES:
            row, col = self.square_to_cell(sq)
            x0, y0 = self.ox + col * size, self.oy + row * size
            self.canvas.coords(self.rect_ids[sq], x0, y0, x0 + size, y0 + size)
            cx, cy = x0 + size // 2, y0 + size // 2
            piece_y = cy + int(size * 0.10)
            self.canvas.coords(self.text_ids[sq], cx, piece_y)
            self.canvas.coords(self.shadow_ids[sq], cx + shadow, piece_y + shadow)
            self.canvas.coords(self.num_ids[sq], cx, y0 + int(size * 0.13))

    def flip_board(self):
        self.flipped = not self.flipped
        self.layout_board()

    def on_canvas_click(self, event):
        # Human always plays White. Ignore clicks while the AI is thinking or
        # whenever it is Black's turn.
        if self.outcome is not None or self.ai_thinking or self.board.turn != chess.WHITE:
            return
        col = (event.x - self.ox) // self.sq_size
        row = (event.y - self.oy) // self.sq_size
        if 0 <= row < 8 and 0 <= col < 8:
            self.handle_square(self.cell_to_square(row, col))

    def handle_square(self, square):
        piece = self.board.piece_at(square)
        own_piece = piece is not None and piece.color == chess.WHITE

        if self.selected_square is None:
            if own_piece:
                self.select_square(square)
            return
        if square == self.selected_square:
            self.clear_selection()
            return
        if own_piece:
            self.select_square(square)
            return

        candidates = [
            move for move in self.moves_from.get(self.selected_square, ())
            if move.to_square == square
        ]
        if not candidates:
            self.show_status("Illegal move. Choose a highlighted square.")
            return

        move = candidates[0]
        if len(candidates) > 1:
            promo = chess.QUEEN if self.auto_queen else self.ask_promotion()
            move = next(m for m in candidates if m.promotion == promo)

        self.board.push(move)
        self.after_position_change()
        if self.outcome is None:
            self.ai_thinking = True
            self.show_status("AI is thinking...")
            self.root.after(120, self.make_ai_move)

    def ask_promotion(self):
        choice = {"value": chess.QUEEN}
        win = tk.Toplevel(self.root)
        win.title("Pawn Promotion")
        win.transient(self.root)
        win.resizable(False, False)
        tk.Label(win, text="Promote pawn to:", padx=12, pady=8).pack()
        row = tk.Frame(win)
        row.pack(padx=12, pady=(0, 12))
        for name, piece_type in (
            ("Queen", chess.QUEEN), ("Rook", chess.ROOK),
            ("Bishop", chess.BISHOP), ("Knight", chess.KNIGHT),
        ):
            def pick(p=piece_type):
                choice["value"] = p
                win.destroy()
            tk.Button(row, text=name, width=8, command=pick).pack(side="left", padx=3)
        win.grab_set()
        self.root.wait_window(win)
        return choice["value"]

    def make_ai_move(self):
        if self.outcome is not None or self.board.turn != chess.BLACK:
            self.ai_thinking = False
            return
        try:
            move = choose_ai_move(self.board, self.ai_depth)
            if move is not None:
                self.board.push(move)
        finally:
            self.ai_thinking = False
        self.after_position_change()

    def select_square(self, square):
        self.selected_square = square
        self.legal_targets = {m.to_square for m in self.moves_from.get(square, ())}
        self.update_squares()
        self.show_status(self.base_status if self.legal_targets else "That piece has no legal moves.")

    def clear_selection(self):
        self.selected_square = None
        self.legal_targets = set()
        self.update_squares()
        self.show_status(self.base_status)

    def after_position_change(self):
        self.selected_square = None
        self.legal_targets = set()
        self.matrix = board_to_numeric_matrix(self.board)
        self.moves_from = {}
        for move in self.board.legal_moves:
            self.moves_from.setdefault(move.from_square, []).append(move)
        self.outcome = self.board.outcome()
        self.base_status = self.compute_status()
        self.update_squares()
        self.show_status(self.base_status)
        self.refresh_matrix_window()
        if self.outcome is not None and not self.result_shown:
            self.result_shown = True
            self.root.after(50, self.show_game_result)

    def update_squares(self):
        selected_piece = self.board.piece_at(self.selected_square) if self.selected_square is not None else None
        pawn_selected = selected_piece is not None and selected_piece.piece_type == chess.PAWN
        for sq in chess.SQUARES:
            code = self.code_at(sq)
            if sq == self.selected_square:
                bg = SELECTED_SQUARE
            elif sq in self.legal_targets:
                en_passant = pawn_selected and sq == self.board.ep_square
                bg = CAPTURE_SQUARE if (code != 0 or en_passant) else LEGAL_SQUARE
            elif (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 0:
                bg = DARK_SQUARE
            else:
                bg = LIGHT_SQUARE
            state = (code, bg)
            if self.drawn.get(sq) == state:
                continue
            self.drawn[sq] = state
            self.canvas.itemconfig(self.rect_ids[sq], fill=bg)
            piece = self.board.piece_at(sq)
            if piece is None:
                symbol, number = "", ""
            else:
                symbol = PIECE_SYMBOLS[piece.piece_type]
                number = f"{code:g}"
            # White pieces are bright with a shadow; Black pieces are dark.
            fill = "#FFFFFF" if code > 0 else "#111111"
            self.canvas.itemconfig(self.text_ids[sq], text=symbol, fill=fill)
            self.canvas.itemconfig(self.shadow_ids[sq], text=symbol if code > 0 else "")
            self.canvas.itemconfig(self.num_ids[sq], text=number)

    def compute_status(self):
        if self.board.is_checkmate():
            winner = "AI (Black)" if self.board.turn == chess.WHITE else "You (White)"
            return f"Checkmate! {winner} wins."
        if self.outcome is not None:
            reason = self.outcome.termination.name.replace("_", " ").lower()
            return f"Draw: {reason}."
        if self.board.is_check():
            return f"{self.turn_name()} to move — CHECK!"
        return f"{self.turn_name()} to move."

    def turn_name(self):
        return "You (White)" if self.board.turn == chess.WHITE else "AI (Black)"

    def show_status(self, text):
        self.status_label.config(text=text)

    def new_game(self):
        self.board.reset()
        self.ai_thinking = False
        self.result_shown = False
        self.after_position_change()

    def undo(self):
        if not self.board.move_stack:
            return
        # Undo the human move and the AI response, if present.
        self.ai_thinking = False
        self.board.pop()
        if self.board.move_stack and self.board.turn == chess.BLACK:
            self.board.pop()
        self.result_shown = False
        self.after_position_change()

    def show_matrix(self):
        if self.matrix_window is not None and self.matrix_window.winfo_exists():
            self.matrix_window.lift()
            self.refresh_matrix_window()
            return
        window = tk.Toplevel(self.root)
        window.title("Numeric Board Matrix")
        legend = (
            "Numeric representation (8 x 8):\n"
            "0.0 = empty; positive = White; negative = Black\n"
            "pawn 1.0, knight 3.0, bishop 3.25, rook 5.0, queen 9.0, king 10.0"
        )
        tk.Label(window, text=legend, justify="left", padx=10, pady=8).pack(anchor="w")
        text_box = tk.Text(window, width=56, height=8, font=("Courier New", 12))
        text_box.pack(padx=10, pady=8, fill="both", expand=True)
        self.matrix_window, self.matrix_text = window, text_box
        self.refresh_matrix_window()

    def refresh_matrix_window(self):
        if self.matrix_window is None or not self.matrix_window.winfo_exists():
            return
        matrix_text = "\n".join(" ".join(f"{value:>6.2f}" for value in row) for row in self.matrix)
        self.matrix_text.config(state="normal")
        self.matrix_text.delete("1.0", "end")
        self.matrix_text.insert("1.0", matrix_text)
        self.matrix_text.config(state="disabled")

    def show_game_result(self):
        outcome = self.outcome
        if outcome is None:
            return
        reason = outcome.termination.name.replace("_", " ").lower()
        if outcome.winner is None:
            result = f"Game over: draw ({reason})."
        else:
            winner = "You (White)" if outcome.winner == chess.WHITE else "AI (Black)"
            result = f"Game over: {winner} wins ({reason})."
        messagebox.showinfo("Game Over", result)


def run_basic_tests():
    board = chess.Board()
    matrix = board_to_numeric_matrix(board)
    assert len(matrix) == 8 and all(len(row) == 8 for row in matrix)
    assert matrix[1] == [-1.0] * 8
    assert matrix[6] == [1.0] * 8
    assert sum(value != 0 for row in matrix for value in row) == 32
    assert chess.Move.from_uci("e2e4") in board.legal_moves
    assert chess.Move.from_uci("e2e5") not in board.legal_moves
    assert choose_ai_move(board, depth=1) in board.legal_moves
    print("All basic tests passed.")


def parse_args():
    parser = argparse.ArgumentParser(description="Human vs AI chess with numeric pieces.")
    parser.add_argument("--size", type=int, default=None, help="initial square size in pixels")
    parser.add_argument("--auto-queen", action="store_true", help="promote to queen without asking")
    parser.add_argument("--depth", type=int, default=2, choices=(1, 2, 3),
                        help="AI search depth: 1 easy, 2 normal, 3 stronger but slower")
    parser.add_argument("--test", action="store_true", help="run basic checks and exit")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.test:
        run_basic_tests()
        return
    root = tk.Tk()
    ChessGame(root, square_size=args.size, ai_depth=args.depth, auto_queen=args.auto_queen)
    root.mainloop()


if __name__ == "__main__":
    main()
