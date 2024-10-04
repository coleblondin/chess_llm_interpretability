# %%
import model_setup

import pickle
import sys

model = model_setup.model
t = model_setup.torch

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

########Below code is for running the entire script from terminal rather than interactively
########It's annoying to do it that way because it reloads the entire model every time
# pgn = sys.argv[1]
# pgn = "1.e4"
# if len(sys.argv) > 2:
#     num_ply = int(sys.argv[2])
# else:
#     num_ply = 1
# if pgn[0] != ";":
#     pgn = ";" + pgn 
# if pgn[-1] != " ":
#     pgn = pgn + " "
# MAX_TOKENS = 1000

# token_count = 0
# ply_count = 0

# while ply_count < num_ply and token_count < MAX_TOKENS:
#     next_token = decode(model(t.tensor(encode(pgn))).argmax(dim=-1).tolist()[0])[-1]
#     pgn = pgn + next_token
#     if next_token == "#":
#         break
#     if next_token == " ":
#         ply_count += 1
#     token_count += 1

# print(pgn)
# %%
def play_n_moves(model: model_setup.HookedTransformer, pgn: str, n: int = 1) -> str:
    if pgn[0] != ";":
        pgn = ";" + pgn 
    if pgn[-1] != " ":
        pgn = pgn + " "
    while n > 0:
        next_token = decode(model(t.tensor(encode(pgn))).argmax(dim=-1).tolist()[0])[-1]
        pgn = pgn + next_token
        if next_token == " ":
            n -= 1
    return pgn
# %%

def strip_spaces_after_periods(s: str) -> str:
    out = s[0]
    for i in range(1, len(s)):
        if s[i] != " " or s[i-1] != ".":
            out += s[i]
    return out
# %%
pgn=play_n_moves(model=model, pgn=";1.e4 e5")
print(pgn)
# %%

pgn=play_n_moves(model=model, pgn=strip_spaces_after_periods(";1. Na3 Na6 2. Rb1 Rb8 3. Nc4 Nc5 4. Na5 Na4 5. Nb3 Nb6 6. Na1 Na8 "))
print(pgn)


# %%
pgn=play_n_moves(
    model=model, 
    # pgn=strip_spaces_after_periods(";"),
    # pgn=strip_spaces_after_periods(";1. Na3 Na6 2. Rb1 Rb8 3. Nc4 Nc5 4. Na5 Na4 5. Nb3 Nb6 6. Na1 Na8 "),
    # pgn=strip_spaces_after_periods(";1. Na3 Na6 2. Rb1 Rb8 3. Nc4 Nc5 4. Na5 Na4 5. Nb3 Nb6 6. Na1 Na8 7. e4"),
    pgn=strip_spaces_after_periods(";1. Na3 Na6 2. Rb1 Rb8 3. Nc4 Nc5 4. Na5 Na4 5. Nb3 Nb6 6. Na1 Na8 7. e4 e5 "),
    # pgn=strip_spaces_after_periods(";1. Na3 Na6 2. Rb1 Rb8 3. Nc4 Nc5 4. Na5 Na4 5. Nb3 Nb6 6. Na1 Na8 7. e4 e5 8. Nf3 "),
    # pgn=strip_spaces_after_periods(";1. Na3 Na6 2. Rb1 Rb8 3. Nc4 Nc5 4. Na5 Na4 5. Nb3 Nb6 6. Na1 Na8 7. e4 e5 8. d4 "),
    # pgn=strip_spaces_after_periods(";1. Na3 Na6 2. Rb1 Rb8 3. Nc4 Nc5 4. Na5 Na4 5. Nb3 Nb6 6. Na1 Na8 7. e4 e5 8. Nf3 d5 9. exd5 Qxd5 10. d4"),
    # pgn=strip_spaces_after_periods(";1. Na3 Na6 2. Rb1 Rb8 3. Nc4 Nc5 4. Na5 Na4 5. Nb3 Nb6 6. Na1 Na8 7. e4 e5 8. Nf3 d5 9. exd5 Qxd5 10. d4 e4 11. c4 Qf5 12. Nd2 Nf6 13. Nc2"),
    n=30
)
print(pgn)
