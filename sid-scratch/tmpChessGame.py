# # %%
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import chess
# import chess.pgn
# import io


# # %%

# import model_setup

# import pickle
# import sys

# file_path = '../train_ChessGPT/data/lichess_hf_dataset/meta.pkl'
# with open(file_path, 'rb') as f:
#     data = pickle.load(f)

# itos = data['itos']
# stoi = data['stoi']

# def decode(l):
#     return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# def encode(s):
#     out = []
#     for i in list(s):
#         out.append(stoi[i])
#     return out
# model = model_setup.model






# # %%
# def create_board(board):
#     """Create a plotly figure representing the chess board."""
#     fig = make_subplots(rows=8, cols=8)
    
#     for row in range(8):
#         for col in range(8):
#             square = chess.square(col, 7-row)
#             piece = board.piece_at(square)
            
#             if piece:
#                 symbol = chess.UNICODE_PIECE_SYMBOLS[piece.symbol().lower()]
#                 color = 'white' if piece.color == chess.WHITE else 'black'
#                 fig.add_annotation(
#                     x=col, y=row,
#                     text=symbol,
#                     showarrow=False,
#                     font=dict(size=40, color=color)
#                 )
    
#     fig.update_xaxes(range=[-0.5, 7.5], showgrid=False, zeroline=False, visible=False)
#     fig.update_yaxes(range=[-0.5, 7.5], showgrid=False, zeroline=False, visible=False)
    
#     fig.update_layout(
#         width=600, height=600,
#         plot_bgcolor='navajowhite',
#         showlegend=False
#     )
    
#     return fig

# def update_board(fig, board):
#     """Update the plotly figure with the current board state."""
#     fig.data = []
#     fig.layout.annotations = []
    
#     for row in range(8):
#         for col in range(8):
#             square = chess.square(col, 7-row)
#             piece = board.piece_at(square)
            
#             if piece:
#                 symbol = chess.UNICODE_PIECE_SYMBOLS[piece.symbol().lower()]
#                 color = 'white' if piece.color == chess.WHITE else 'black'
#                 fig.add_annotation(
#                     x=col, y=row,
#                     text=symbol,
#                     showarrow=False,
#                     font=dict(size=40, color=color)
#                 )

# def get_square_from_click(x, y):
#     """Convert click coordinates to chess square."""
#     file = int(x + 0.5)
#     rank = 7 - int(y + 0.5)
#     return chess.square(file, rank)

# class ChessGame:
#     def __init__(self, other_player):
#         self.board = chess.Board()
#         self.fig = create_board(self.board)
#         self.selected_square = None
#         self.other_player = other_player
#         self.game = chess.pgn.Game()
#         self.node = self.game
    
#     def on_click(self, trace, points, state):
#         if len(points.point_inds) == 0:
#             return
        
#         x, y = points.xs[0], points.ys[0]
#         square = get_square_from_click(x, y)
        
#         if self.selected_square is None:
#             piece = self.board.piece_at(square)
#             if piece and piece.color == self.board.turn:
#                 self.selected_square = square
#         else:
#             move = chess.Move(self.selected_square, square)
#             if move in self.board.legal_moves:
#                 self.make_move(move)
#                 self.update_display()
                
#                 # Other player's turn
#                 self.other_player_move()
            
#             self.selected_square = None
    
#     def make_move(self, move):
#         self.board.push(move)
#         self.node = self.node.add_variation(move)
    
#     def other_player_move(self):
#         pgn = str(self.game)
#         move_uci = self.other_player(pgn)
#         move = chess.Move.from_uci(move_uci)
#         self.make_move(move)
#         self.update_display()
    
#     def update_display(self):
#         update_board(self.fig, self.board)
#         self.fig.show()
    
#     def play(self):
#         self.fig.show()



# # %%
# def play_n_moves(model: model_setup.HookedTransformer, pgn: str, n: int = 1) -> str:
#     if pgn[0] != ";":
#         pgn = ";" + pgn 
#     if pgn[-1] != " ":
#         pgn = pgn + " "
#     next_move = ""
#     while n > 0:
#         next_token = decode(model(t.tensor(encode(pgn))).argmax(dim=-1).tolist()[0])[-1]
#         next_move += next_token
#         pgn = pgn + next_token
#         if next_token == " ":
#             n -= 1
#     return pgn
# def play_next_move(model: model_setup.HookedTransformer, pgn: str) -> str:
#     if pgn[0] != ";":
#         pgn = ";" + pgn 
#     if pgn[-1] != " ":
#         pgn = pgn + " "
#     next_move = ""
#     n=1
#     while n > 0:
#         next_token = decode(model(t.tensor(encode(pgn))).argmax(dim=-1).tolist()[0])[-1]
#         next_move += next_token
#         pgn = pgn + next_token
#         if next_token == " ":
#             n -= 1
#     return next_move

# def strip_spaces_after_periods(s: str) -> str:
#     out = s[0]
#     for i in range(1, len(s)):
#         if s[i] != " " or s[i-1] != ".":
#             out += s[i]
#     return out

# # Example usage:
# def other_player(pgn):
#     if 1:
#         # This is a dummy function that just makes a random move
#         game = chess.pgn.read_game(io.StringIO(pgn))
#         board = game.end().board()
#         return str(list(board.legal_moves)[0])  # Return first legal move
#     else:
#         strip_spaces_after_periods(pgn)
#         play_next_move(pgn)
#         return


# # %%
# game = ChessGame(other_player)
# game.fig.data[0].on_click(game.on_click)
# game.play()
















# %%

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import chess
import chess.pgn
import io
from ipywidgets import Output
from IPython.display import display
import plotly.io as pio
pio.renderers.default = "notebook"

def create_board(board):
    """Create a plotly figure representing the chess board."""
    fig = make_subplots(rows=8, cols=8)
    
    for row in range(8):
        for col in range(8):
            color = 'lightgrey' if (row + col) % 2 == 0 else 'white'
            fig.add_trace(go.Scatter(
                x=[col, col+1, col+1, col, col],
                y=[row, row, row+1, row+1, row],
                fill="toself",
                fillcolor=color,
                line=dict(color='black'),
                showlegend=False,
                hoverinfo='none'
            ))
    
    update_pieces(fig, board)
    
    fig.update_xaxes(range=[-0.5, 8.5], showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[-0.5, 8.5], showgrid=False, zeroline=False, visible=False)
    
    fig.update_layout(
        width=600, height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )
    
    return fig

def update_pieces(fig, board):
    """Update the plotly figure with the current piece positions."""
    for row in range(8):
        for col in range(8):
            square = chess.square(col, 7-row)
            piece = board.piece_at(square)
            
            if piece:
                symbol = chess.UNICODE_PIECE_SYMBOLS[piece.symbol().lower()]
                color = 'white' if piece.color == chess.WHITE else 'black'
                fig.add_annotation(
                    x=col+0.5, y=row+0.5,
                    text=symbol,
                    showarrow=False,
                    font=dict(size=40, color=color)
                )

def get_square_from_click(x, y):
    """Convert click coordinates to chess square."""
    file = int(x)
    rank = 7 - int(y)
    return chess.square(file, rank)

class ChessGame:
    def __init__(self, other_player):
        self.board = chess.Board()
        self.fig = go.FigureWidget(create_board(self.board))
        self.selected_square = None
        self.other_player = other_player
        self.game = chess.pgn.Game()
        self.node = self.game
        self.output = Output()
        
        self.fig.data[0].on_click(self.on_click)
    
    def on_click(self, trace, points, state):
        with self.output:
            if len(points.xs) == 0:
                return
            
            x, y = points.xs[0], points.ys[0]
            square = get_square_from_click(x, y)
            
            if self.selected_square is None:
                piece = self.board.piece_at(square)
                if piece and piece.color == self.board.turn:
                    self.selected_square = square
                    print(f"Selected {piece} at {chess.SQUARE_NAMES[square]}")
            else:
                move = chess.Move(self.selected_square, square)
                if move in self.board.legal_moves:
                    self.make_move(move)
                    self.update_display()
                    print(f"Moved to {chess.SQUARE_NAMES[square]}")
                    
                    if not self.board.is_game_over():
                        print("AI is thinking...")
                        self.other_player_move()
                else:
                    print("Invalid move")
                
                self.selected_square = None
    
    def make_move(self, move):
        self.board.push(move)
        self.node = self.node.add_variation(move)
        
        if self.board.is_game_over():
            result = self.board.result()
            print(f"Game over. Result: {result}")
    
    def other_player_move(self):
        pgn = str(self.game)
        move_uci = self.other_player(pgn)
        move = chess.Move.from_uci(move_uci)
        self.make_move(move)
        self.update_display()
        print(f"AI moved: {move}")
    
    def update_display(self):
        self.fig.data = []
        self.fig.layout.annotations = []
        update_pieces(self.fig, self.board)
    
    def play(self):
        display(self.fig)
        self.fig.show()
        display(self.output)






# %%
        
import model_setup

import pickle
import sys

file_path = '../train_ChessGPT/data/lichess_hf_dataset/meta.pkl'
with open(file_path, 'rb') as f:
    data = pickle.load(f)

itos = data['itos']
stoi = data['stoi']

def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

def encode(s):
    out = []
    for i in list(s):
        out.append(stoi[i])
    return out
model = model_setup.model

def play_n_moves(model: model_setup.HookedTransformer, pgn: str, n: int = 1) -> str:
    if pgn[0] != ";":
        pgn = ";" + pgn 
    if pgn[-1] != " ":
        pgn = pgn + " "
    next_move = ""
    while n > 0:
        next_token = decode(model(t.tensor(encode(pgn))).argmax(dim=-1).tolist()[0])[-1]
        next_move += next_token
        pgn = pgn + next_token
        if next_token == " ":
            n -= 1
    return pgn
def play_next_move(model: model_setup.HookedTransformer, pgn: str) -> str:
    if pgn[0] != ";":
        pgn = ";" + pgn 
    if pgn[-1] != " ":
        pgn = pgn + " "
    next_move = ""
    n=1
    while n > 0:
        next_token = decode(model(t.tensor(encode(pgn))).argmax(dim=-1).tolist()[0])[-1]
        next_move += next_token
        pgn = pgn + next_token
        if next_token == " ":
            n -= 1
    return next_move

def strip_spaces_after_periods(s: str) -> str:
    out = s[0]
    for i in range(1, len(s)):
        if s[i] != " " or s[i-1] != ".":
            out += s[i]
    return out

# Example usage:
def other_player(pgn):
    if 1:
        # This is a dummy function that just makes a random move
        game = chess.pgn.read_game(io.StringIO(pgn))
        board = game.end().board()
        return str(list(board.legal_moves)[0])  # Return first legal move
    else:
        strip_spaces_after_periods(pgn)
        play_next_move(pgn)
        return



# %%
game = ChessGame(other_player)
game.play()
# %%























# %%

import plotly.graph_objects as go
import chess
import chess.pgn
import io
from IPython.display import display, clear_output

def create_board(board):
    """Create a plotly figure representing the chess board."""
    fig = go.Figure()
    
    # Create the chessboard
    for row in range(8):
        for col in range(8):
            color = 'lightgrey' if (row + col) % 2 == 0 else 'white'
            fig.add_shape(
                type="rect",
                x0=col, y0=row, x1=col+1, y1=row+1,
                line=dict(color="Black"),
                fillcolor=color
            )
    
    # Add pieces
    for row in range(8):
        for col in range(8):
            square = chess.square(col, 7-row)
            piece = board.piece_at(square)
            if piece:
                symbol = chess.UNICODE_PIECE_SYMBOLS[piece.symbol().lower()]
                color = 'black' if piece.color == chess.BLACK else 'white'
                fig.add_annotation(
                    x=col+0.5, y=row+0.5,
                    text=symbol,
                    showarrow=False,
                    font=dict(size=30, color=color)
                )
    
    fig.update_xaxes(range=[0, 8], showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[0, 8], showgrid=False, zeroline=False, visible=False)
    fig.update_layout(width=600, height=600, margin=dict(l=0, r=0, t=0, b=0))
    
    return fig

class ChessGame:
    def __init__(self, other_player):
        self.board = chess.Board()
        self.other_player = other_player
        self.game = chess.pgn.Game()
        self.node = self.game
    
    def make_move(self, move):
        self.board.push(move)
        self.node = self.node.add_variation(move)
        
        if self.board.is_game_over():
            result = self.board.result()
            print(f"Game over. Result: {result}")
    
    def other_player_move(self):
        pgn = str(self.game)
        move_uci = self.other_player(pgn)
        move = chess.Move.from_uci(move_uci)
        self.make_move(move)
        print(f"AI moved: {move}")
    
    def play(self):
        while not self.board.is_game_over():
            clear_output(wait=True)
            fig = create_board(self.board)
            fig.show()
            
            if self.board.turn == chess.WHITE:
                move_uci = input("Enter your move (in UCI format, e.g. 'e2e4'): ")
                try:
                    move = chess.Move.from_uci(move_uci)
                    if move in self.board.legal_moves:
                        self.make_move(move)
                    else:
                        print("Illegal move. Try again.")
                        continue
                except ValueError:
                    print("Invalid input. Try again.")
                    continue
            else:
                self.other_player_move()
        
        clear_output(wait=True)
        fig = create_board(self.board)
        fig.show()
        print(f"Game over. Result: {self.board.result()}")





# %%
        
import model_setup

import pickle
import sys

file_path = '../train_ChessGPT/data/lichess_hf_dataset/meta.pkl'
with open(file_path, 'rb') as f:
    data = pickle.load(f)

itos = data['itos']
stoi = data['stoi']

def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

def encode(s):
    out = []
    for i in list(s):
        out.append(stoi[i])
    return out
model = model_setup.model

def play_n_moves(model: model_setup.HookedTransformer, pgn: str, n: int = 1) -> str:
    if pgn[0] != ";":
        pgn = ";" + pgn 
    if pgn[-1] != " ":
        pgn = pgn + " "
    next_move = ""
    while n > 0:
        next_token = decode(model(t.tensor(encode(pgn))).argmax(dim=-1).tolist()[0])[-1]
        next_move += next_token
        pgn = pgn + next_token
        if next_token == " ":
            n -= 1
    return pgn
def play_next_move(model: model_setup.HookedTransformer, pgn: str) -> str:
    if pgn[0] != ";":
        pgn = ";" + pgn 
    if pgn[-1] != " ":
        pgn = pgn + " "
    next_move = ""
    n=1
    while n > 0:
        next_token = decode(model(t.tensor(encode(pgn))).argmax(dim=-1).tolist()[0])[-1]
        next_move += next_token
        pgn = pgn + next_token
        if next_token == " ":
            n -= 1
    return next_move

def strip_spaces_after_periods(s: str) -> str:
    out = s[0]
    for i in range(1, len(s)):
        if s[i] != " " or s[i-1] != ".":
            out += s[i]
    return out

# Example usage:
def other_player(pgn):
    if 1:
        # This is a dummy function that just makes a random move
        game = chess.pgn.read_game(io.StringIO(pgn))
        board = game.end().board()
        return str(list(board.legal_moves)[0])  # Return first legal move
    else:
        strip_spaces_after_periods(pgn)
        play_next_move(pgn)
        return



# %%
game = ChessGame(other_player)
# while True:
game.play()
# %%