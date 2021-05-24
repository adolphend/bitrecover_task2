# BER calculation for QAM symbol vectors
# Use QAM_BER_gray fucntion to calculate BER

def Gray_alphabet(bm):
    import numpy as np

    gseq = np.empty((2 ** bm, bm), dtype=int)
    for i in range(2 ** bm):
        buf = i ^ (i >> 1)
        buf = np.asarray([int(x) for x in bin(buf)[2:]])
        gseq[i, :] = np.append(np.zeros(bm - buf.size, dtype=int), buf)
    return gseq


def Gray_QAM_bit_abc(m):
    import numpy as np
    bm = int(m / 2)
    gseq = Gray_alphabet(bm)
    gabc = np.concatenate((np.tile(gseq, reps=(2 ** bm, 1)), np.repeat(gseq, repeats=2 ** bm, axis=0)), axis=1)
    return gabc


def Gray_QAM_sym_abc(m, norm=True):
    import numpy as np

    ms = int(np.sqrt(2 ** m))
    abc_side = np.arange(0, ms) * 2 - (ms - 1)
    QAM_abc = np.tile(abc_side, reps=ms) + 1j * np.repeat(np.flip(abc_side, axis=0), repeats=ms, axis=0)
    if norm:
        QAM_abc = QAM_abc / np.std(QAM_abc)
    return QAM_abc


def hard_slice(QAMsyms, m, norm=True):
    import numpy as np

    alphabet = Gray_QAM_sym_abc(m, norm)
    sym_indices = list(map(lambda sym: np.argmin(np.abs(sym - alphabet)), QAMsyms))
    return alphabet[sym_indices], sym_indices


def QAM2gray_bits(QAMsyms, QAM_order, norm=True):
    # Converts vector QAM complex-valued symbols to the Gray coded bits
    # QAMsyms - QAM symbol vector to convert
    # QAM_order - order of the QAM target alphabet (e.g. 16 for 16QAM)
    # norm - whether the targer QAM alphabet has unitary power
    import numpy as np

    m = np.log2(QAM_order)  # Number of bits per QAM symbol

    # Popular error tracking
    if np.mod(m, 1.) != 0.:
        raise ValueError('Given QAM order should be some power of 2.')
    if np.mod(m, 2.) != 0.:
        raise ValueError('Non-square constellations are not supported (e.g. 32QAM, 128QAM)')
    if QAMsyms.ndim != 1:
        raise ValueError('Input array of QAM symbols must be an array')

    m = int(m)  # Convert bit number to integer after checking its value
    QAM_indices = hard_slice(QAMsyms, m, norm)[1]  # Hard slice the input QAM sequence and return its
    bit_alphabet = Gray_QAM_bit_abc(m)  # Bit patterns, corresponding to every symbol from QAM alphabet
    bit_seq = np.concatenate(tuple((bit_alphabet[QAM_ind] for QAM_ind in QAM_indices)), axis=0)
    return bit_seq


def QAM_BER_gray(QAMsyms_chk, QAMsyms_ref, QAM_order, norm=True):
    # Calculates BER between the two QAM symbol vectors in input data
    # QAMsyms - QAM symbol vector to convert
    # QAM_order - order of the QAM target alphabet (e.g. 16 for 16QAM)
    # norm - whether the targer QAM alphabet has unitary power
    import numpy as np

    bits_chk = QAM2gray_bits(QAMsyms_chk, QAM_order, norm)
    bits_ref = QAM2gray_bits(QAMsyms_ref, QAM_order, norm)
    BER = np.mean(np.logical_xor(bits_ref, bits_chk))
    return BER
