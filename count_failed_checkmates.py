# %%
import model_setup
import pickle
import sys
model = model_setup.model
t = model_setup.torch
import chess

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
import pandas as pd
import re
# datafile = "data/lichess_test.csv"
datafile = "data_variable_string_lengths/lichess_test.csv"
df = pd.read_csv(datafile)


# %%
n=0
total_checks_possibe = 0
total_checks_predicted = 0
total_checks_missed = 0
total_checkmates_possibe = 0
total_checkmates_predicted = 0
total_checkmates_predicted_as_check = 0
total_checkmates_missed = 0
for game in df['transcript']:
    n+=1
    # print(game)
    # print('--------')
    if '+' in game:
        check_inds = [m.start() for m in re.finditer('\+', game)]
        for ind in check_inds:
            total_checks_possibe += 1
            # print(game[:ind])
            # print('--------')
            next_token = decode(model(t.tensor(encode(game[:ind]))).argmax(dim=-1).tolist()[0])[-1]
            if next_token == '+':
                total_checks_predicted += 1
            else:
                total_checks_missed += 1
            # print(next_token)
            # print('--------')
    if '#' in game:
        check_inds = [m.start() for m in re.finditer('#', game)]
        for ind in check_inds:
            total_checkmates_possibe += 1
            # print(game[:ind])
            # print('--------')
            next_token = decode(model(t.tensor(encode(game[:ind]))).argmax(dim=-1).tolist()[0])[-1]
            if next_token == '#':
                total_checkmates_predicted += 1
            elif next_token == '+':
                total_checkmates_predicted_as_check += 1
            else:
                total_checkmates_missed += 1
    #         print(next_token)
    #         print('--------')
    # print('-----------------------------------------------------------------------------------------------')
    # pgn=play_n_moves
    print_every = 200
    if (n % print_every) == print_every-1:
        print(f"{total_checks_possibe=}")
        print(f"{total_checks_predicted=}")
        print(f"{total_checks_missed=}")
        print(f"{total_checkmates_possibe=}")
        print(f"{total_checkmates_predicted=}")
        print(f"{total_checkmates_predicted_as_check=}")
        print(f"{total_checkmates_missed=}")
        print('---------------------------------------------------------------')
    # if n> 1000:
    #     break

# %%

print(f"{total_checks_possibe=}")
print(f"{total_checks_predicted=}")
print(f"{total_checkmates_possibe=}")
print(f"{total_checkmates_predicted=}")
print(f"{total_checkmates_predicted_as_check=}")




###################################
######### OUTPUTS BELOW ###########
###################################

# total_checks_possibe=1266
# total_checks_predicted=1263
# total_checks_missed=3
# total_checkmates_possibe=46
# total_checkmates_predicted=42
# total_checkmates_predicted_as_check=4
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=2621
# total_checks_predicted=2616
# total_checks_missed=5
# total_checkmates_possibe=87
# total_checkmates_predicted=77
# total_checkmates_predicted_as_check=10
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=3918
# total_checks_predicted=3912
# total_checks_missed=6
# total_checkmates_possibe=136
# total_checkmates_predicted=122
# total_checkmates_predicted_as_check=14
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=5330
# total_checks_predicted=5321
# total_checks_missed=9
# total_checkmates_possibe=180
# total_checkmates_predicted=164
# total_checkmates_predicted_as_check=16
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=6763
# total_checks_predicted=6752
# total_checks_missed=11
# total_checkmates_possibe=231
# total_checkmates_predicted=209
# total_checkmates_predicted_as_check=22
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=8223
# total_checks_predicted=8210
# total_checks_missed=13
# total_checkmates_possibe=281
# total_checkmates_predicted=256
# total_checkmates_predicted_as_check=25
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=9662
# total_checks_predicted=9648
# total_checks_missed=14
# total_checkmates_possibe=339
# total_checkmates_predicted=309
# total_checkmates_predicted_as_check=30
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=11031
# total_checks_predicted=11014
# total_checks_missed=17
# total_checkmates_possibe=392
# total_checkmates_predicted=358
# total_checkmates_predicted_as_check=34
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=12462
# total_checks_predicted=12444
# total_checks_missed=18
# total_checkmates_possibe=442
# total_checkmates_predicted=406
# total_checkmates_predicted_as_check=36
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=13839
# total_checks_predicted=13819
# total_checks_missed=20
# total_checkmates_possibe=479
# total_checkmates_predicted=439
# total_checkmates_predicted_as_check=40
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=15315
# total_checks_predicted=15290
# total_checks_missed=25
# total_checkmates_possibe=534
# total_checkmates_predicted=487
# total_checkmates_predicted_as_check=47
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=16755
# total_checks_predicted=16729
# total_checks_missed=26
# total_checkmates_possibe=583
# total_checkmates_predicted=529
# total_checkmates_predicted_as_check=54
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=18202
# total_checks_predicted=18172
# total_checks_missed=30
# total_checkmates_possibe=643
# total_checkmates_predicted=586
# total_checkmates_predicted_as_check=57
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=19739
# total_checks_predicted=19706
# total_checks_missed=33
# total_checkmates_possibe=678
# total_checkmates_predicted=616
# total_checkmates_predicted_as_check=62
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=21159
# total_checks_predicted=21121
# total_checks_missed=38
# total_checkmates_possibe=731
# total_checkmates_predicted=667
# total_checkmates_predicted_as_check=64
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=22452
# total_checks_predicted=22410
# total_checks_missed=42
# total_checkmates_possibe=770
# total_checkmates_predicted=704
# total_checkmates_predicted_as_check=66
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=23799
# total_checks_predicted=23756
# total_checks_missed=43
# total_checkmates_possibe=829
# total_checkmates_predicted=756
# total_checkmates_predicted_as_check=73
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=25149
# total_checks_predicted=25101
# total_checks_missed=48
# total_checkmates_possibe=882
# total_checkmates_predicted=806
# total_checkmates_predicted_as_check=76
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=26509
# total_checks_predicted=26456
# total_checks_missed=53
# total_checkmates_possibe=922
# total_checkmates_predicted=843
# total_checkmates_predicted_as_check=79
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=27840
# total_checks_predicted=27785
# total_checks_missed=55
# total_checkmates_possibe=966
# total_checkmates_predicted=885
# total_checkmates_predicted_as_check=81
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=29205
# total_checks_predicted=29148
# total_checks_missed=57
# total_checkmates_possibe=1014
# total_checkmates_predicted=930
# total_checkmates_predicted_as_check=84
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=30648
# total_checks_predicted=30588
# total_checks_missed=60
# total_checkmates_possibe=1061
# total_checkmates_predicted=970
# total_checkmates_predicted_as_check=91
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=32108
# total_checks_predicted=32048
# total_checks_missed=60
# total_checkmates_possibe=1112
# total_checkmates_predicted=1015
# total_checkmates_predicted_as_check=97
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=33476
# total_checks_predicted=33413
# total_checks_missed=63
# total_checkmates_possibe=1167
# total_checkmates_predicted=1067
# total_checkmates_predicted_as_check=100
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=34808
# total_checks_predicted=34741
# total_checks_missed=67
# total_checkmates_possibe=1214
# total_checkmates_predicted=1112
# total_checkmates_predicted_as_check=102
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=36200
# total_checks_predicted=36131
# total_checks_missed=69
# total_checkmates_possibe=1264
# total_checkmates_predicted=1158
# total_checkmates_predicted_as_check=106
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=37551
# total_checks_predicted=37480
# total_checks_missed=71
# total_checkmates_possibe=1311
# total_checkmates_predicted=1198
# total_checkmates_predicted_as_check=113
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=38949
# total_checks_predicted=38874
# total_checks_missed=75
# total_checkmates_possibe=1360
# total_checkmates_predicted=1239
# total_checkmates_predicted_as_check=121
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=40312
# total_checks_predicted=40234
# total_checks_missed=78
# total_checkmates_possibe=1403
# total_checkmates_predicted=1280
# total_checkmates_predicted_as_check=123
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=41720
# total_checks_predicted=41638
# total_checks_missed=82
# total_checkmates_possibe=1455
# total_checkmates_predicted=1329
# total_checkmates_predicted_as_check=126
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=43142
# total_checks_predicted=43057
# total_checks_missed=85
# total_checkmates_possibe=1504
# total_checkmates_predicted=1374
# total_checkmates_predicted_as_check=130
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=44494
# total_checks_predicted=44408
# total_checks_missed=86
# total_checkmates_possibe=1544
# total_checkmates_predicted=1410
# total_checkmates_predicted_as_check=134
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=45909
# total_checks_predicted=45817
# total_checks_missed=92
# total_checkmates_possibe=1584
# total_checkmates_predicted=1448
# total_checkmates_predicted_as_check=136
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=47384
# total_checks_predicted=47288
# total_checks_missed=96
# total_checkmates_possibe=1628
# total_checkmates_predicted=1489
# total_checkmates_predicted_as_check=139
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=48902
# total_checks_predicted=48803
# total_checks_missed=99
# total_checkmates_possibe=1677
# total_checkmates_predicted=1535
# total_checkmates_predicted_as_check=142
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=50353
# total_checks_predicted=50253
# total_checks_missed=100
# total_checkmates_possibe=1721
# total_checkmates_predicted=1576
# total_checkmates_predicted_as_check=145
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=51825
# total_checks_predicted=51719
# total_checks_missed=106
# total_checkmates_possibe=1772
# total_checkmates_predicted=1623
# total_checkmates_predicted_as_check=149
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=53385
# total_checks_predicted=53275
# total_checks_missed=110
# total_checkmates_possibe=1824
# total_checkmates_predicted=1674
# total_checkmates_predicted_as_check=150
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=54831
# total_checks_predicted=54714
# total_checks_missed=117
# total_checkmates_possibe=1869
# total_checkmates_predicted=1714
# total_checkmates_predicted_as_check=155
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=56237
# total_checks_predicted=56120
# total_checks_missed=117
# total_checkmates_possibe=1918
# total_checkmates_predicted=1759
# total_checkmates_predicted_as_check=159
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=57682
# total_checks_predicted=57556
# total_checks_missed=126
# total_checkmates_possibe=1971
# total_checkmates_predicted=1809
# total_checkmates_predicted_as_check=162
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=59092
# total_checks_predicted=58963
# total_checks_missed=129
# total_checkmates_possibe=2019
# total_checkmates_predicted=1852
# total_checkmates_predicted_as_check=167
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=60511
# total_checks_predicted=60380
# total_checks_missed=131
# total_checkmates_possibe=2059
# total_checkmates_predicted=1890
# total_checkmates_predicted_as_check=169
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=61765
# total_checks_predicted=61633
# total_checks_missed=132
# total_checkmates_possibe=2105
# total_checkmates_predicted=1932
# total_checkmates_predicted_as_check=173
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=63165
# total_checks_predicted=63031
# total_checks_missed=134
# total_checkmates_possibe=2149
# total_checkmates_predicted=1968
# total_checkmates_predicted_as_check=181
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=64558
# total_checks_predicted=64420
# total_checks_missed=138
# total_checkmates_possibe=2189
# total_checkmates_predicted=2006
# total_checkmates_predicted_as_check=183
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=65985
# total_checks_predicted=65846
# total_checks_missed=139
# total_checkmates_possibe=2232
# total_checkmates_predicted=2046
# total_checkmates_predicted_as_check=186
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=67415
# total_checks_predicted=67271
# total_checks_missed=144
# total_checkmates_possibe=2284
# total_checkmates_predicted=2092
# total_checkmates_predicted_as_check=192
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=68841
# total_checks_predicted=68696
# total_checks_missed=145
# total_checkmates_possibe=2321
# total_checkmates_predicted=2127
# total_checkmates_predicted_as_check=194
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=70188
# total_checks_predicted=70042
# total_checks_missed=146
# total_checkmates_possibe=2360
# total_checkmates_predicted=2160
# total_checkmates_predicted_as_check=200
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=71638
# total_checks_predicted=71488
# total_checks_missed=150
# total_checkmates_possibe=2409
# total_checkmates_predicted=2207
# total_checkmates_predicted_as_check=202
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=73127
# total_checks_predicted=72974
# total_checks_missed=153
# total_checkmates_possibe=2457
# total_checkmates_predicted=2248
# total_checkmates_predicted_as_check=209
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=74648
# total_checks_predicted=74494
# total_checks_missed=154
# total_checkmates_possibe=2491
# total_checkmates_predicted=2281
# total_checkmates_predicted_as_check=210
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=76062
# total_checks_predicted=75907
# total_checks_missed=155
# total_checkmates_possibe=2530
# total_checkmates_predicted=2316
# total_checkmates_predicted_as_check=214
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=77509
# total_checks_predicted=77353
# total_checks_missed=156
# total_checkmates_possibe=2578
# total_checkmates_predicted=2360
# total_checkmates_predicted_as_check=218
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=78994
# total_checks_predicted=78838
# total_checks_missed=156
# total_checkmates_possibe=2624
# total_checkmates_predicted=2404
# total_checkmates_predicted_as_check=220
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=80430
# total_checks_predicted=80273
# total_checks_missed=157
# total_checkmates_possibe=2672
# total_checkmates_predicted=2446
# total_checkmates_predicted_as_check=226
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=81900
# total_checks_predicted=81740
# total_checks_missed=160
# total_checkmates_possibe=2721
# total_checkmates_predicted=2491
# total_checkmates_predicted_as_check=230
# total_checkmates_missed=0
# ---------------------------------------------------------------
# total_checks_possibe=83365
# total_checks_predicted=83204
# total_checks_missed=161
# total_checkmates_possibe=2773
# total_checkmates_predicted=2536
# total_checkmates_predicted_as_check=237
# total_checkmates_missed=0
# ---------------------------------------------------------------
