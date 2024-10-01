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

PARSED_PGN_INDEX = 5
DATA_DIR = "/root/chess_llm_interpretability/data/"
file_path = DATA_DIR + "lichess_db_chess960_rated_2024-08_with_parsed.csv"
with open(file_path) as fp:
    reader = csv.reader(fp, delimiter=",", quotechar='"')
    next(reader, None)  # skip the headers
    data_read = [row for row in reader]

# %%

pgns = [data_read[i][PARSED_PGN_INDEX] for i in range(5) if data_read[i][PARSED_PGN_INDEX] != ""]

file_path = '../train_ChessGPT/data/lichess_hf_dataset/meta.pkl'
with open(file_path, 'rb') as f:
    meta = pickle.load(f)

# %%

total_pgns = len(pgns)

for pgn in pgns:
    board = chess_utils.pgn_string_to_board(pgn)
    encoded_input = chess_utils.encode_string(meta, pgn)
    model_input = t.tensor(encoded_input).unsqueeze(0).to(DEVICE)
    move = chess_utils.get_model_move(model, meta, model_input)
    print(move)
# %%
