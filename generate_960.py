# %%
# This came from ChatGPT, don't @me if it breaks...
import chess
import chess.pgn
import chess.engine
import random

# Path to your Stockfish engine (update this path to wherever Stockfish is located on your system)
STOCKFISH_PATH = "/usr/games/stockfish"


# # %%
# start_pos = random.randint(0, 959)  # There are 960 possible Chess960 starting positions
# board = chess.Board(chess960=True)
# board.set_chess960_pos(start_pos)
# game = chess.pgn.Game()
# with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
#     node = game
#     # Generate moves using Stockfish
#     for i in range(random.randint(10, 40)):  # Random length game between 10 to 40 moves
#         if board.is_game_over():
#             break
#         # Stockfish evaluates the best move
#         result = engine.play(board, chess.engine.Limit(time=0.1))  # Limit time per move to 0.1 seconds
#         board.push(result.move)
#         node = node.add_variation(result.move)
#         # Check for game end (checkmate, stalemate, draw conditions)
#         if board.is_checkmate():
#             game.headers["Result"] = "1-0" if board.turn == chess.BLACK else "0-1"
#             break
#         elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves():
#             game.headers["Result"] = "1/2-1/2"
#             break
#     else:
#         # If game doesn't end early, it's a draw
#         game.headers["Result"] = "1/2-1/2"


# %%
# Function to generate a Chess960 game using Stockfish
def generate_chess960_game_with_engine(player1_name, player2_name, player1_elo, player2_elo):
    # Initialize a Chess960 game with a random starting position
    start_pos = random.randint(0, 959)  # There are 960 possible Chess960 starting positions
    board = chess.Board(chess960=True)
    board.set_chess960_pos(start_pos)
    
    # Create a new PGN game
    game = chess.pgn.Game()
    
    # Set the headers with metadata
    game.headers["Event"] = "Chess960"
    game.headers["White"] = player1_name
    game.headers["Black"] = player2_name
    game.headers["WhiteElo"] = str(player1_elo)
    game.headers["BlackElo"] = str(player2_elo)
    game.headers["Variant"] = "Chess960"
    game.headers["FEN"] = board.fen()
    
    # Open the Stockfish engine
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        node = game
        
        # Generate moves using Stockfish
        for i in range(random.randint(10, 40)):  # Random length game between 10 to 40 moves
            if board.is_game_over():
                break
            
            # Stockfish evaluates the best move
            result = engine.play(board, chess.engine.Limit(time=0.1))  # Limit time per move to 0.1 seconds
            board.push(result.move)
            node = node.add_variation(result.move)
            
            # Check for game end (checkmate, stalemate, draw conditions)
            if board.is_checkmate():
                game.headers["Result"] = "1-0" if board.turn == chess.BLACK else "0-1"
                break
            elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves():
                game.headers["Result"] = "1/2-1/2"
                break
        else:
            # If game doesn't end early, it's a draw
            game.headers["Result"] = "1/2-1/2"
    
    # Return PGN string and metadata
    return str(game), {
        'player1': player1_name,
        'player2': player2_name,
        'player1_elo': player1_elo,
        'player2_elo': player2_elo,
        'result': game.headers["Result"],
        'starting_position': start_pos
    }

# Example of generating multiple Chess960 games and storing them
def generate_chess960_games_database_with_engine(num_games):
    games_data = []
    for _ in range(num_games):
        # Example player names and ELOs
        player1_name = "Player1"
        player2_name = "Player2"
        player1_elo = random.randint(1600, 2500)
        player2_elo = random.randint(1600, 2500)
        
        # Generate game and its metadata using Stockfish
        pgn_game, metadata = generate_chess960_game_with_engine(player1_name, player2_name, player1_elo, player2_elo)
        
        # Store both the PGN string and metadata
        games_data.append({
            'pgn': pgn_game,
            'metadata': metadata
        })
    
    return games_data

# Example usage: Generate a database of 5 games using Stockfish
chess960_games_db = generate_chess960_games_database_with_engine(5)

# Print the PGN and metadata for the first game
print("PGN of first game:\n", chess960_games_db[0]['pgn'])
print("\nMetadata of first game:\n", chess960_games_db[0]['metadata'])

# %%
