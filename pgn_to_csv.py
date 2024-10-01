# %%
# if 0:
import chess
import chess.pgn
import csv


class AllMovedVisitor(chess.pgn.BaseVisitor):
    def __init__(self):
        # self.pgn = None
        self.piece_moved = {}
        self.ply_count = 0
        self.all_pieces_ply = None

        #There's definitely some cute Pythonic way to do this
        for sq in chess.SQUARES:
            if sq < 8 or sq > 55:
                self.piece_moved[sq] = False
    
    def visit_move(self, board: chess.Board, move: chess.Move) -> None:
        # if self.pgn is None:
        if self.all_pieces_ply is None:
            self.ply_count += 1
            if move.from_square in self.piece_moved:
                self.piece_moved[move.from_square] = True
            if move.to_square in self.piece_moved:
                self.piece_moved[move.to_square] = True
            
            if all(moved is True for moved in self.piece_moved.values()):
                self.all_pieces_ply = self.ply_count

    
    def result(self) -> int:
        return self.all_pieces_ply

def get_all_pieces_moved_ply(game: chess.pgn.Game) -> int:
    vis = AllMovedVisitor()
    return game.accept(vis)

def strip_spaces_after_periods(s: str) -> str:
    out = s[0]
    for i in range(1, len(s)):
        if s[i] != " " or s[i-1] != ".":
            out += s[i]
    return out

#Assumes spaces after periods have already been removed
def parse_first_n_ply(s: str, n: int) -> str:
    out = s[0]
    for i in range(1, len(s)):
        if s[i] == " ":
            n -= 1
            if n <= 0:
                break
        out += s[i]
    return out

# %%
# Input PGN file
DATA_DIR = "/root/chess_llm_interpretability/data/"
pgn_file = DATA_DIR+"lichess_db_chess960_rated_2024-04.pgn"

# Output CSV file
# csv_file = pgn_file[:-3] + "csv"
csv_file = pgn_file[:-4] + "_with_parsed.csv"

# Define CSV headers
headers = ["WhiteElo", "BlackElo", "Result", "StartingFEN", "Transcript", "TranscriptMoved"]

# Open the PGN file and CSV file
with open(pgn_file) as pgn, open(csv_file, mode='w', newline='') as csv_out:
    writer = csv.DictWriter(csv_out, fieldnames=headers)
    writer.writeheader()  # Write the CSV headers
    
    # Parse each game from the PGN file
    while True:
        game = chess.pgn.read_game(pgn)
        if game is None:
            break  # No more games in the PGN file
        
        # Extract required data from the game headers
        white_elo = game.headers.get("WhiteElo", "")
        black_elo = game.headers.get("BlackElo", "")
        result = game.headers.get("Result", "")
        starting_fen = game.headers.get("FEN", "")
        # transcript = str(game)  # PGN transcript of the game
        moves = str(game.mainline().__repr__()).split('(')[-1].split(')')[0]
        if moves == "":
            continue
        transcript = ';' + strip_spaces_after_periods(moves)

        transcript_all_pieces_moved = None
        all_pieces_moved_ply_count = get_all_pieces_moved_ply(game)

        if all_pieces_moved_ply_count is not None:
            transcript_all_pieces_moved = parse_first_n_ply(transcript, all_pieces_moved_ply_count)

        # Write the game data to the CSV
        writer.writerow({
            "WhiteElo": white_elo.strip('"'),
            "BlackElo": black_elo.strip('"'),
            "Result": result.strip('"'),
            "StartingFEN": starting_fen.strip('"'),
            "Transcript": transcript.strip('"'),
            "TranscriptMoved": transcript_all_pieces_moved.strip('"') if transcript_all_pieces_moved is not None else ""
        })

print("Conversion to CSV complete.")
# elif 0:
#     import chess.pgn
#     import csv

#     # Input PGN file
#     DATA_DIR = "/root/chess_llm_interpretability/data/"
#     pgn_file = DATA_DIR + "lichess_db_chess960_rated_2024-07.pgn"

#     # Output CSV file
#     csv_file = pgn_file[:-3] + "csv"

#     # Define CSV headers
#     headers = ["WhiteElo", "BlackElo", "Result", "transcript"]

#     # Open the PGN file and CSV file
#     with open(pgn_file) as pgn, open(csv_file, mode='w', newline='') as csv_out:
#         writer = csv.DictWriter(csv_out, fieldnames=headers)
#         writer.writeheader()  # Write the CSV headers
        
#         # Collect all rows to write at once
#         rows = []
#         # n=0
#         # Parse each game from the PGN file
#         while True:
#             game = chess.pgn.read_game(pgn)
#             if game is None:
#                 break  # No more games in the PGN file
            
#             # Extract required data from the game headers
#             white_elo = game.headers.get("WhiteElo", "").strip('"')
#             black_elo = game.headers.get("BlackElo", "").strip('"')
#             result = game.headers.get("Result", "").strip('"')
#             transcript = ';' + str(game.mainline().__repr__()).split('(')[-1].split(')')[0]
            
#             # Append the game data to the rows list
#             rows.append({
#                 "WhiteElo": white_elo,
#                 "BlackElo": black_elo,
#                 "Result": result,
#                 "transcript": transcript.strip('"')
#             })
#             # n+=1
#         # Write all rows to the CSV at once
#         writer.writerows(rows)

#     print("Conversion to CSV complete.")
# else:
#     import chess.pgn
#     import csv
#     import gc  # Garbage collector to free memory

#     # Input PGN file
#     DATA_DIR = "/root/chess_llm_interpretability/data/"
#     pgn_file = DATA_DIR + "lichess_db_chess960_rated_2024-06.pgn"

#     # Output CSV file
#     csv_file = pgn_file[:-3] + "csv"

#     # Define CSV headers
#     headers = ["WhiteElo", "BlackElo", "Result", "transcript"]

#     # Batch size (e.g., write to file every 1000 games)
#     batch_size = 1000

#     # Open the PGN file and CSV file
#     with open(pgn_file) as pgn, open(csv_file, mode='w', newline='') as csv_out:
#         writer = csv.DictWriter(csv_out, fieldnames=headers)
#         writer.writeheader()  # Write the CSV headers
        
#         rows = []  # To collect batch of rows
#         game_count = 0  # Counter to keep track of games processed
        
#         # Parse each game from the PGN file
#         while True:
#             game = chess.pgn.read_game(pgn)
#             if game is None:
#                 break  # No more games in the PGN file
            
#             # Extract required data from the game headers
#             white_elo = game.headers.get("WhiteElo", "").strip('"')
#             black_elo = game.headers.get("BlackElo", "").strip('"')
#             result = game.headers.get("Result", "").strip('"')
#             transcript = ';' + str(game.mainline().__repr__()).split('(')[-1].split(')')[0]
            
#             # Add the game data to the current batch of rows
#             rows.append({
#                 "WhiteElo": white_elo,
#                 "BlackElo": black_elo,
#                 "Result": result,
#                 "transcript": transcript.strip('"')
#             })
            
#             game_count += 1
            
#             # If we've reached the batch size, write to file and reset rows
#             if game_count % batch_size == 0:
#                 writer.writerows(rows)
#                 rows.clear()  # Clear the batch to free up memory
#                 gc.collect()  # Force garbage collection to free memory
        
#         # Write any remaining rows that were not written in the last batch
#         if rows:
#             writer.writerows(rows)

#     print(f"Conversion to CSV complete. Total games processed: {game_count}")

# %%
