# %%
import csv
import chess_utils
import pickle
import model_setup
# from pgn_to_csv import DATA_DIR

model = model_setup.model
t = model_setup.torch
DEVICE = (
    "cuda" if t.cuda.is_available() else "mps" if t.backends.mps.is_available() else "cpu"
)

# %%

data_read= []

FEN_INDEX = 3
PARSED_PGN_INDEX = 5
DATA_DIR = "/root/chess_llm_interpretability/data/"
file_path = DATA_DIR + "lichess_db_chess960_rated_2024-08_with_parsed.csv"
with open(file_path) as fp:
    reader = csv.reader(fp, delimiter=",", quotechar='"')
    next(reader, None)  # skip the headers
    data_read = [row for row in reader]

# %%

test_games = []
count = 0
for row in data_read:
    if row[PARSED_PGN_INDEX] != "":
        test_games.append((row[FEN_INDEX], row[PARSED_PGN_INDEX]))
        count += 1
    if count >= 10:
        break

file_path = '../train_ChessGPT/data/lichess_hf_dataset/meta.pkl'
with open(file_path, 'rb') as f:
    meta = pickle.load(f)

# %%

total_games = len(test_games)
correct_count = 0

for ind, game in enumerate(test_games):
    fen = game[0]
    pgn = game[1]
    board = chess_utils.pgn_string_to_board(pgn, fen=fen, chess960=True)
    done = False
    while not done:
        pgn += " "
        encoded_input = chess_utils.encode_string(meta, pgn)
        model_input = t.tensor(encoded_input).unsqueeze(0).to(DEVICE)
        model_output = chess_utils.get_model_move(model, meta=meta, idx=model_input, temperature=0.0)
        move = model_output
        print(ind)
        print(model_output)
        if ";" in move or "#" in move:
            print(f"Game end in {ind}")
            done = True
            move = move.split(";")[0]
        if "." in move:
            move = move.split(".")[1]
        try:
            board.push_san(move)
            pgn += model_output
            print(pgn)
            print(pgn[-1] == " ")
            if done:
                correct_count += 1
                print(f"Successful at index {ind}")
        except:
            break

        

print(correct_count / total_games)
        
# %%
