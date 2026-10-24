# ---------------------------------------------------------------------------
# Amino-acid encoding dictionaries
# ---------------------------------------------------------------------------
# Integer states used throughout the project (q = 21): 1 = gap / ambiguous,
# 2..21 = the 20 canonical residues in alphabetical one-letter order.
# Same mapping as `FluStudy.AA_TO_INT`.

using BioSequences: AminoAcid, AA_Gap, AA_A, AA_C, AA_D, AA_E, AA_F, AA_G, AA_H, AA_I, AA_K, AA_L, AA_M,
                    AA_N, AA_P, AA_Q, AA_R, AA_S, AA_T, AA_V, AA_W, AA_Y, AA_Term, AA_B, AA_J, AA_X, AA_Z

const intToaa  = Dict(1=>AA_Gap, 2=>AA_A, 3=>AA_C, 4=>AA_D, 5=>AA_E, 6=>AA_F,
                      7=>AA_G, 8=>AA_H, 9=>AA_I, 10=>AA_K, 11=>AA_L, 12=>AA_M,
                      13=>AA_N, 14=>AA_P, 15=>AA_Q, 16=>AA_R, 17=>AA_S, 18=>AA_T,
                      19=>AA_V, 20=>AA_W, 21=>AA_Y)

const charToaa = Dict('-'=>AA_Gap, 'A'=>AA_A, 'C'=>AA_C, 'D'=>AA_D, 'E'=>AA_E,
                      'F'=>AA_F, 'G'=>AA_G, 'H'=>AA_H, 'I'=>AA_I, 'K'=>AA_K,
                      'L'=>AA_L, 'M'=>AA_M, 'N'=>AA_N, 'P'=>AA_P, 'Q'=>AA_Q,
                      'R'=>AA_R, 'S'=>AA_S, 'T'=>AA_T, 'V'=>AA_V, 'W'=>AA_W,
                      'Y'=>AA_Y)

const aaToint  = Dict(AA_Term=>1, AA_B=>1, AA_J=>1, AA_X=>1, AA_Z=>1,
                      AA_Gap=>1, AA_A=>2, AA_C=>3, AA_D=>4, AA_E=>5, AA_F=>6,
                      AA_G=>7, AA_H=>8, AA_I=>9, AA_K=>10, AA_L=>11, AA_M=>12,
                      AA_N=>13, AA_P=>14, AA_Q=>15, AA_R=>16, AA_S=>17, AA_T=>18,
                      AA_V=>19, AA_W=>20, AA_Y=>21)
