# %%

# Lobotomy time

import matplotlib.pyplot as plt
import numpy as np
from sid_chess_utils import INT_TO_PIECE
import torch
import model_setup
import pickle
from fancy_einsum import einsum
from functools import partial
import chess_utils
import chess
# import sys


SAVED_PROBE_DIR = "linear_probes/good_piece_probes/"
DEVICE = "cuda"


## Load model ##
model = model_setup.model
t = model_setup.torch
# model.reset_hooks()

file_path = '../train_ChessGPT/data/lichess_hf_dataset/meta.pkl'
with open(file_path, 'rb') as f:
    META = pickle.load(f)

itos = META['itos']
stoi = META['stoi']

def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

def encode(s):
    out = []
    for i in list(s):
        out.append(stoi[i])
    return out


#######################
## Load piece probes ##
#######################
probes = {}
for layer in range(8):
    probe_to_test = f"tf_lens_lichess_8layers_ckpt_no_optimizer_chess_piece_probe_layer_{layer}.pth"
    probe_file_location = f"{SAVED_PROBE_DIR}{probe_to_test}"
    # Load the probe and run it on an example!
    checkpoint = torch.load(probe_file_location, map_location=DEVICE)
    linear_probe_MDRRC = checkpoint["linear_probe"]
    probes[layer] = linear_probe_MDRRC


#######################
## Load skill probes ##
#######################
skill_probes = {}
for layer in range(8):
    SKILL_PROBE_DIR = "linear_probes/good_skill_probes/"
    state_dict_name = f"tf_lens_lichess_8layers_ckpt_no_optimizer_chess_skill_probe_layer_{layer}.pth"
    checkpoint = torch.load(f"{SKILL_PROBE_DIR}{state_dict_name}", map_location=DEVICE)
    skill_probes[layer] = checkpoint["linear_probe"]
    # average_high_elo = skill_probes[layer][layer]["average_high_elo_activation"]
    # average_low_elo = skill_probes[layer]["average_low_elo_activation"]
    # difference_vector = skill_probes[layer]["difference_vector"]
    # skill_probes[layer]["difference_vector"] = difference_vector

    # new_state_dict_name = f"type=probe_model=8layers_layer={layer}.pt"
    # torch.save(state_dict, new_state_dict_name)


# model.reset_hooks()
my_pgn=";e4"
logits, cache = model.run_with_cache(torch.tensor([encode(my_pgn)]).to(DEVICE)[:, :], return_type='logits')
resid_post_BLD = cache["resid_post", 5][
        :, :
    ]  # shape (batch_size, pgn_str_length - 1, d_model)



probe_out_MBLRRC = einsum(
    "batch pos d_model, modes d_model rows cols options -> modes batch pos rows cols options",
    resid_post_BLD,
    linear_probe_MDRRC,
)
probe_log_probs_MBLRRC = probe_out_MBLRRC.log_softmax(-1).detach().cpu().numpy()
# for i in range(probe_log_probs_MBLRRC.shape[-1]):
#     plt.figure(figsize=(4,4))
#     # plt.imshow(np.exp(probe_log_probs_MBLRRC[0,0,0,:,:,i]), cmap='Grays')
#     plt.imshow(np.exp(probe_log_probs_MBLRRC[0,0,0,:,:,:]).argmax(axis=-1))
#     plt.colorbar()
#     plt.title(f"Probs of being {INT_TO_PIECE[i]}")
#     plt.show()
#     plt.close()


def strip_spaces_after_periods(s: str) -> str:
    out = s[0]
    for i in range(1, len(s)):
        if s[i] != " " or s[i-1] != ".":
            out += s[i]
    return out

MAGIC_PGN_QUEENSIDE = ";1. Na3 Na6 2. Rb1 Rb8 3. Nc4 Nc5 4. Na5 Na4 5. Nb3 Nb6 6. Na1 Na8 "
MAGIC_PGN_KINGSIDE = ";-5. Nh3 Nh6 -4. Rg1 Rg8 -3. Nf4 Nf5 -2. Nh5 Nh4 -1. Ng3 Ng6 0. Nh1 Nh8 "

if 0:
    orig_pgn = ";1.e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 "
    # Wants to move ???
    # ninesixty_piece_index = original_piece_index+1
    # scale = ???
    # Now wants to move ???
elif 0:
    orig_pgn = ';1. b3 Nf6 2. Bb2 e6 3. Bxf6 Qxf6 4. d4 d5 5. e3 c5 6. Nf3 cxd4 7. Be2 Nc6 8. exd4 e5 9. dxe5 Nxe5 10. Nbd2 Bd6 11. O-O O-O 12. Nxe5 Qxe5 13. Nf3 Qf6 14. Bd3 Bd7 15. h3 Bc6 16. Re1 Rae8 17. Qd2 Bf4 18. Qb4 '#Bd6 19. Qd4 Qxd4 20. Nxd4 Bc5 21. Nxc6 bxc6 22. Rad1 Bb4 23. Rxe8 Rxe8 24. Kf1 g6 25. g3 Kf8 26. c4 f5 27. cxd5 cxd5 28. Bb5 Rd8 29. Rd4 Bc3 30. Rd3 d4 31. Bc4 Re8 32. b4 Re1+ 33. Kg2 Rb1 34. b5 Rb2 35. a4 Be1 36. Rf3 Ke7 37. Rf4 Bc3 38. h4 Kf6 39. a5 Ke5 40. a6 Rb4 41. Bd3 Rb2 42. Rf3 h6 43. h5 gxh5 44. Rxf5+ Kd6 45. Rxh5 Be1 46. Rxh6+ Kc5 47. Rf6 Rxf2+ 48. Rxf2 Bxf2 49. Kxf2 Kb4 50. Kf3 Kc5 51. Ke4 Kb6 52. g4 Ka5 53. g5 Kb6 54. g6 Ka5 55. g7 Kb6 56. g8=Q Ka5 57. Qc4 Kb6 58. Qxd4+ Ka5 59. Qxa7 Kb4 60. Qb8 Ka5 61. a7 Ka4 62. a8=Q+ Kb4 63. Qab7 Kc3 64. Qb6 Kd2 65. Q6b7 Ke1 66. Q7a7 Kd2 67. b6 Kd1 68. b7 Kd2 69. Bc4 Ke1 70. Kd5'
    # ninesixty_piece_index = 6
elif 0:
    orig_pgn = ";1. "
    # ninesixty_piece_index = original_piece_index+1
elif 0:
    orig_pgn = ";1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 "
    # Wants to move Na5
elif 0:
    orig_pgn = MAGIC_PGN_QUEENSIDE
elif 1:
    orig_pgn = MAGIC_PGN_QUEENSIDE + "7. e4 e5 "
else:
    assert False

# orig_pgn = ";1.e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 "
orig_pgn = strip_spaces_after_periods(orig_pgn)
orig_board = chess_utils.pgn_string_to_board(orig_pgn)
encoded_input = chess_utils.encode_string(META, orig_pgn)
model_input = torch.tensor(encoded_input).to(DEVICE).unsqueeze(0)
# model.reset_hooks()
orig_board_argmax_model_move = chess_utils.get_model_move(
    model, META, model_input, temperature=0.0
)
print(f"Model wants to move: {orig_board_argmax_model_move.split('.')[-1]}")
from_square = orig_board.parse_san(orig_board_argmax_model_move.split('.')[-1]).from_square
r = from_square // 8
c = from_square % 8
# orig_board.parse_san(orig_board_argmax_model_move)






# %%


# This is the intervention function. In it, we obtain a vector to flip the square to blank in the model's activations at a given layer
# Multiply it by some scale factor, then subtract it from the model's activations
# If we make this function more modular and pass all variables in (probes, r, c, etc), it is much slower
def flip_hook(
    resid,  # shape is (1, num_white_moves, d_model)
    hook,
    r,
    c,
    layer: int,
    original_piece_index: int,
    ninesixty_piece_index: int,
    scale: float = 0.1,
):
    # print(f"Lobotomizing at layer {layer}, hook {hook.name}")
    
    target = 0.0
    original_piece_probe = probes[layer][:, :, r, c, original_piece_index].squeeze()
    ninesixty_piece_probe = probes[layer][:, :, r, c, ninesixty_piece_index].squeeze()

    flip_dir = (original_piece_probe) - (ninesixty_piece_probe)
    flip_dir = flip_dir / flip_dir.norm()

    resid[0, :] -= scale * flip_dir

    # For experimentation with dynamic scale setting
    # coeff = resid[0, move_of_interest_index] @ flip_dir / flip_dir.norm()

    # So we only print once during inference
    # if resid.shape[1] <= move_of_interest_index + 1:
    #     print(
    #         f"Layer: {layer}, coeff: {coeff:10.3f}, scale: {scale:10.3f}, target: {target:10.3f}"
    #     )

# Step 6: Intervene on the model's activations and get the model move under the modified board state
# model.reset_hooks()
fwd_hooks = []
scale=1.0
for layer in probes:
    # if layer != 6:
    #     continue
    original_piece_index = chess_utils.PIECE_TO_ONE_HOT_MAPPING[chess_utils.PIECE_TO_INT[orig_board.piece_at(from_square).piece_type]]
    # ninesixty_piece_index = original_piece_index-1
    ninesixty_piece_index = original_piece_index+2
    temp_hook_fn = partial(flip_hook, r=r, c=c, layer=layer, original_piece_index=original_piece_index, ninesixty_piece_index=ninesixty_piece_index, scale=scale)
    hook_name = f"blocks.{layer}.hook_resid_post"
    fwd_hooks.append((hook_name, temp_hook_fn))

ADD_SKILL=False
if ADD_SKILL:
    def make_it_good_as_shit_hook(
        resid,  # shape is (1, num_white_moves, d_model)
        hook,
        layer: int,
        how_good_do_you_want_it_to_be: int=10,
        how_good_do_you_think_it_was: int=1,
        scale: float = 0.1,
    ):
        print(f"Lobotomizing at layer {layer}, hook {hook.name}")
        
        target = 0.0
        original_piece_probe = probes[layer][:, :, 0, 0, how_good_do_you_think_it_was].squeeze()
        ninesixty_piece_probe = probes[layer][:, :, 0, 0, how_good_do_you_want_it_to_be].squeeze()

        flip_dir = (original_piece_probe) - (ninesixty_piece_probe)
        flip_dir = flip_dir / flip_dir.norm()

        resid[0, :] -= scale * flip_dir
    for layer in probes:
        # if layer != 6:
        #     continue
        scale=1.0
        temp_hook_fn = partial(make_it_good_as_shit_hook, layer=layer, how_good_do_you_want_it_to_be=10, how_good_do_you_think_it_was=0, scale=scale)
        hook_name = f"blocks.{layer}.hook_resid_post"
    fwd_hooks.append((hook_name, temp_hook_fn))


orig_board_argmax_lobotomized_model_move = chess_utils.get_model_move(
    model, META, model_input, temperature=0.0, fwd_hooks=fwd_hooks
)
print(f"Now model wants to move: {orig_board_argmax_lobotomized_model_move}")









# %%

orig_pgn = MAGIC_PGN_QUEENSIDE + "7.e4 e5 "
orig_pgn = MAGIC_PGN_QUEENSIDE + "7.e4 c5 "
# orig_pgn = MAGIC_PGN_KINGSIDE + "1.e4 e5 2.Nc3 Nc6 "
orig_pgn = strip_spaces_after_periods(orig_pgn)

# for scale in np.arange(10)/5:
# for scale in [0.45, 0.5, 0.55]:
for scale in (t.arange(10)/100 + 0.3):
    pieces_to_replace = {
        (0,0):chess_utils.PIECE_TO_INT[chess.KNIGHT], 
        (0,1):chess_utils.PIECE_TO_INT[chess.ROOK], 
        (7,0):chess_utils.PIECE_TO_INT[chess.KNIGHT], 
        (7,1):chess_utils.PIECE_TO_INT[chess.ROOK]
    }
    replacements = {
        (0,0):chess_utils.PIECE_TO_INT[chess.ROOK], 
        (0,1):chess_utils.PIECE_TO_INT[chess.KNIGHT], 
        (7,0):chess_utils.PIECE_TO_INT[chess.ROOK], 
        (7,1):chess_utils.PIECE_TO_INT[chess.KNIGHT]
    }
    
    pgn = orig_pgn
    # board = chess_utils.pgn_string_to_board(pgn)
    # board = chess_utils.pgn_string_to_board(";1.e4 e5 ")
    board = chess_utils.pgn_string_to_board(";1.e4 c5 ")
    n_turns=100
    print("==============================================================================================================")
    print(f"====================================    SCALE {scale}       ==================================================")
    print("==============================================================================================================")
    try:
        for turn in range(n_turns):
            encoded_input = chess_utils.encode_string(META, pgn)
            model_input = torch.tensor(encoded_input).to(DEVICE).unsqueeze(0)
            fwd_hooks = []
            for (rank,file) in pieces_to_replace.keys():
                for layer in probes:
                    # if layer != 6:
                    #     continue
                    scale=scale
                    temp_hook_fn = partial(flip_hook, layer=layer, r=rank, c=file, original_piece_index=pieces_to_replace[(rank,file)], ninesixty_piece_index=replacements[(rank,file)], scale=scale)
                    hook_name = f"blocks.{layer}.hook_resid_post"
                    fwd_hooks.append((hook_name, temp_hook_fn))
            argmax_model_move = chess_utils.get_model_move(
                model, META, model_input, temperature=0.0, fwd_hooks=fwd_hooks
            )
            pgn += argmax_model_move + " "
            print(f"Model wants to move: {argmax_model_move}")
            board.push_san(argmax_model_move.split('.')[-1])
            # Check if the move changed any of our magic pieces
            from_square = board.peek().from_square
            move_from_rank = from_square // 8
            move_from_file = from_square % 8
            if (move_from_rank, move_from_file) in pieces_to_replace.keys():
                del pieces_to_replace[(move_from_rank, move_from_file)]
                del replacements[(move_from_rank, move_from_file)]
            to_square = board.peek().to_square
            move_to_rank = to_square // 8
            move_to_file = to_square % 8
            if (move_to_rank, move_to_file) in pieces_to_replace.keys():
                del pieces_to_replace[(move_to_rank, move_to_file)]
                del replacements[(move_to_rank, move_to_file)]
            print(f"{pgn=}")
    except:
        print("SAD SAID SADE SED SAHD :( :( :(")
    





# %%



# # %%
# # %%

# # Lobotomy time

# import matplotlib.pyplot as plt
# import numpy as np
# from sid_chess_utils import INT_TO_PIECE
# import torch
# import model_setup
# import pickle
# from fancy_einsum import einsum
# from functools import partial
# import chess_utils
# # import sys

# SAVED_PROBE_DIR = "linear_probes/good_piece_probes/"
# DEVICE = "cuda"

# model = model_setup.model
# t = model_setup.torch
# model.reset_hooks()

# file_path = '../train_ChessGPT/data/lichess_hf_dataset/meta.pkl'
# with open(file_path, 'rb') as f:
#     META = pickle.load(f)

# itos = META['itos']
# stoi = META['stoi']

# def decode(l):
#     return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# def encode(s):
#     out = []
#     for i in list(s):
#         out.append(stoi[i])
#     return out

# probes = {}
# for layer in range(8):
#     probe_to_test = f"tf_lens_lichess_8layers_ckpt_no_optimizer_chess_piece_probe_layer_{layer}.pth"
#     probe_file_location = f"{SAVED_PROBE_DIR}{probe_to_test}"
#     # Load the probe and run it on an example!
#     checkpoint = torch.load(probe_file_location, map_location=DEVICE)
#     linear_probe_MDRRC = checkpoint["linear_probe"]
#     probes[layer] = linear_probe_MDRRC


# model.reset_hooks()
# my_pgn=";e4"
# logits, cache = model.run_with_cache(torch.tensor([encode(my_pgn)]).to(DEVICE)[:, :], return_type='logits')
# resid_post_BLD = cache["resid_post", 5][
#         :, :
#     ]  # shape (batch_size, pgn_str_length - 1, d_model)



# probe_out_MBLRRC = einsum(
#     "batch pos d_model, modes d_model rows cols options -> modes batch pos rows cols options",
#     resid_post_BLD,
#     linear_probe_MDRRC,
# )
# probe_log_probs_MBLRRC = probe_out_MBLRRC.log_softmax(-1).detach().cpu().numpy()
# # for i in range(probe_log_probs_MBLRRC.shape[-1]):
# #     plt.figure(figsize=(4,4))
# #     # plt.imshow(np.exp(probe_log_probs_MBLRRC[0,0,0,:,:,i]), cmap='Grays')
# #     plt.imshow(np.exp(probe_log_probs_MBLRRC[0,0,0,:,:,:]).argmax(axis=-1))
# #     plt.colorbar()
# #     plt.title(f"Probs of being {INT_TO_PIECE[i]}")
# #     plt.show()
# #     plt.close()


# def strip_spaces_after_periods(s: str) -> str:
#     out = s[0]
#     for i in range(1, len(s)):
#         if s[i] != " " or s[i-1] != ".":
#             out += s[i]
#     return out

# if 1:
#     orig_pgn = ";1.e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 "
#     # ninesixty_piece_index = original_piece_index+1
#     # 
# elif 0:
#     orig_pgn = ';1. b3 Nf6 2. Bb2 e6 3. Bxf6 Qxf6 4. d4 d5 5. e3 c5 6. Nf3 cxd4 7. Be2 Nc6 8. exd4 e5 9. dxe5 Nxe5 10. Nbd2 Bd6 11. O-O O-O 12. Nxe5 Qxe5 13. Nf3 Qf6 14. Bd3 Bd7 15. h3 Bc6 16. Re1 Rae8 17. Qd2 Bf4 18. Qb4 '#Bd6 19. Qd4 Qxd4 20. Nxd4 Bc5 21. Nxc6 bxc6 22. Rad1 Bb4 23. Rxe8 Rxe8 24. Kf1 g6 25. g3 Kf8 26. c4 f5 27. cxd5 cxd5 28. Bb5 Rd8 29. Rd4 Bc3 30. Rd3 d4 31. Bc4 Re8 32. b4 Re1+ 33. Kg2 Rb1 34. b5 Rb2 35. a4 Be1 36. Rf3 Ke7 37. Rf4 Bc3 38. h4 Kf6 39. a5 Ke5 40. a6 Rb4 41. Bd3 Rb2 42. Rf3 h6 43. h5 gxh5 44. Rxf5+ Kd6 45. Rxh5 Be1 46. Rxh6+ Kc5 47. Rf6 Rxf2+ 48. Rxf2 Bxf2 49. Kxf2 Kb4 50. Kf3 Kc5 51. Ke4 Kb6 52. g4 Ka5 53. g5 Kb6 54. g6 Ka5 55. g7 Kb6 56. g8=Q Ka5 57. Qc4 Kb6 58. Qxd4+ Ka5 59. Qxa7 Kb4 60. Qb8 Ka5 61. a7 Ka4 62. a8=Q+ Kb4 63. Qab7 Kc3 64. Qb6 Kd2 65. Q6b7 Ke1 66. Q7a7 Kd2 67. b6 Kd1 68. b7 Kd2 69. Bc4 Ke1 70. Kd5'
#     # ninesixty_piece_index = 6
# elif 1:
#     orig_pgn = ";1. "
# else:
#     assert False

# # orig_pgn = ";1.e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 "
# orig_pgn = strip_spaces_after_periods(orig_pgn)
# orig_board = chess_utils.pgn_string_to_board(orig_pgn)
# encoded_input = chess_utils.encode_string(META, orig_pgn)
# model_input = torch.tensor(encoded_input).to(DEVICE).unsqueeze(0)
# model.reset_hooks()
# orig_board_argmax_model_move = chess_utils.get_model_move(
#     model, META, model_input, temperature=0.0
# )
# print(f"Model wants to move: {orig_board_argmax_model_move.split('.')[-1]}")
# from_square = orig_board.parse_san(orig_board_argmax_model_move.split('.')[-1]).from_square
# r = from_square // 8
# c = from_square % 8
# # orig_board.parse_san(orig_board_argmax_model_move)






# # %%
# # The ultimate lobotomy
# orig_board = chess_utils.pgn_string_to_board("1. ")
# model.reset_hooks()
# for r in range(2):
#     for c in range(8):
#         if (r==0) and (c==4):
#             continue
#         def flip_hook(
#             resid,  # shape is (1, num_white_moves, d_model)
#             hook,
#             layer: int,
#             original_piece_index: int,
#             ninesixty_piece_index: int,
#             scale: float = 0.1,
#         ):
#             print(f"Lobotomizing at layer {layer}, hook {hook.name}")
            
#             target = 0.0
#             original_piece_probe = probes[layer][:, :, r, c, original_piece_index].squeeze()
#             ninesixty_piece_probe = probes[layer][:, :, r, c, ninesixty_piece_index].squeeze()

#             flip_dir = (original_piece_probe) - (ninesixty_piece_probe)
#             flip_dir = flip_dir / flip_dir.norm()

#             resid[0, :] -= scale * flip_dir

#             # For experimentation with dynamic scale setting
#             # coeff = resid[0, move_of_interest_index] @ flip_dir / flip_dir.norm()

#             # So we only print once during inference
#             # if resid.shape[1] <= move_of_interest_index + 1:
#             #     print(
#             #         f"Layer: {layer}, coeff: {coeff:10.3f}, scale: {scale:10.3f}, target: {target:10.3f}"
#             #     )

#         # Step 6: Intervene on the model's activations and get the model move under the modified board state
#         scale=0.132
#         for layer in probes:
#             # if layer != 6:
#             #     continue
#             original_piece_index = chess_utils.PIECE_TO_ONE_HOT_MAPPING[chess_utils.PIECE_TO_INT[orig_board.piece_at(from_square).piece_type]]
#             # ninesixty_piece_index = original_piece_index-1
#             ninesixty_piece_index =  6
#             temp_hook_fn = partial(flip_hook, layer=layer, original_piece_index=original_piece_index, ninesixty_piece_index=ninesixty_piece_index, scale=scale)
#             hook_name = f"blocks.{layer}.hook_resid_post"
#             model.add_hook(hook_name, temp_hook_fn)



# orig_board_argmax_lobotomized_model_move = chess_utils.get_model_move(
#     model, META, model_input, temperature=0.0
# )
# print(f"Model wants to move: {orig_board_argmax_lobotomized_model_move}")

# # %%

# %%
