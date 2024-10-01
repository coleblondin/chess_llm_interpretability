





# %%

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import chess
import chess.pgn
import io
from ipywidgets import Output
from IPython.display import display
# import plotly.io as pio
# pio.renderers.default = "notebook"

def create_board(board):
    """Create a plotly figure representing the chess board."""
    # fig = make_subplots(rows=8, cols=8)
    fig = make_subplots(rows=1, cols=1)
    
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
    print(file, rank)
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
        # self.fig.show()
        display(self.output)















# %%


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import chess
import chess.pgn
import io
from ipywidgets import Output
from IPython.display import display

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
                color = 'black' if piece.color == chess.BLACK else 'white'
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
        
        self.fig.on_click(self.on_click)
    
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
        with self.fig.batch_update():
            self.fig.data = []
            self.fig.layout.annotations = []
            update_pieces(self.fig, self.board)
    
    def play(self):
        display(self.fig)
        display(self.output)

# # Example usage:
# def other_player(pgn):
#     # This is a dummy function that just makes a random move
#     game = chess.pgn.read_game(io.StringIO(pgn))
#     board = game.end().board()
#     return str(list(board.legal_moves)[0])  # Return first legal move

# game = ChessGame(other_player)
# game.play()











# %%






# %%

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import chess
import chess.pgn
import io
from ipywidgets import Output
from IPython.display import display
# import plotly.io as pio
# pio.renderers.default = "notebook"

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
        # self.fig.show()
        display(self.output)















# %%


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import chess
import chess.pgn
import io
from ipywidgets import Output
from IPython.display import display

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
                color = 'black' if piece.color == chess.BLACK else 'white'
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
        
        # Set up the click event
        self.fig.data[0].on_click(self.on_click)
    
    def on_click(self, trace, points, selector):
        with self.output:
            if not points.point_inds:
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
        with self.fig.batch_update():
            self.fig.data = []
            self.fig.layout.annotations = []
            update_pieces(self.fig, self.board)
    
    def play(self):
        display(self.fig)
        display(self.output)

# # Example usage:
# def other_player(pgn):
#     # This is a dummy function that just makes a random move
#     game = chess.pgn.read_game(io.StringIO(pgn))
#     board = game.end().board()
#     return str(list(board.legal_moves)[0])  # Return first legal move

# game = ChessGame(other_player)
# game.play()
# %%















# %%

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import chess
import chess.pgn
import io
from ipywidgets import Button, Output, VBox
from IPython.display import display
import asyncio

def create_board(board):
    """Create a plotly figure representing the chess board."""
    fig = go.Figure()
    # fig = make_subplots(rows=1, cols=1)
    
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
                color = 'black' if piece.color == chess.BLACK else 'white'
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
        self.start_button = Button(description="Start Game")
        self.start_button.on_click(self.on_start)
        
        # Set up the click event
        # self.fig.data[0].on_click(self.on_click)
        self.fig.data[0].on_click(self.on_click)
    
    def on_click(self, trace, points, selector):
        if not points.point_inds:
            return
        
        x, y = points.xs[0], points.ys[0]
        square = get_square_from_click(x, y)
        
        if self.selected_square is None:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                self.selected_square = square
                with self.output:
                    print(f"Selected {piece} at {chess.SQUARE_NAMES[square]}")
        else:
            move = chess.Move(self.selected_square, square)
            if move in self.board.legal_moves:
                self.make_move(move)
                self.update_display()
                with self.output:
                    print(f"Moved to {chess.SQUARE_NAMES[square]}")
                
                if not self.board.is_game_over():
                    with self.output:
                        print("AI is thinking...")
                    self.other_player_move()
            else:
                with self.output:
                    print("Invalid move")
            
            self.selected_square = None
    
    def make_move(self, move):
        self.board.push(move)
        self.node = self.node.add_variation(move)
        
        if self.board.is_game_over():
            result = self.board.result()
            with self.output:
                print(f"Game over. Result: {result}")
    
    def other_player_move(self):
        pgn = str(self.game)
        move_uci = self.other_player(pgn)
        move = chess.Move.from_uci(move_uci)
        self.make_move(move)
        self.update_display()
        with self.output:
            print(f"AI moved: {move}")
    
    def update_display(self):
        with self.fig.batch_update():
            self.fig.data = []
            self.fig.layout.annotations = []
            update_pieces(self.fig, self.board)
    
    async def game_loop(self):
        while not self.board.is_game_over():
            await asyncio.sleep(0.1)  # Small delay to prevent blocking
        
        with self.output:
            print("Game has ended. Final position:")
    
    def on_start(self, b):
        self.start_button.disabled = True
        asyncio.create_task(self.game_loop())
    
    def play(self):
        display(VBox([self.fig, self.start_button, self.output]))



# Example usage:
def other_player(pgn):
    if 1:
        # This is a dummy function that just makes a random move
        game = chess.pgn.read_game(io.StringIO(pgn))
        board = game.end().board()
        return str(list(board.legal_moves)[0])  # Return first legal move

game = ChessGame(other_player)
# while True:
game.play()