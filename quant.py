import numpy as np
import struct
import sys
import math
from multiprocessing import Pool

QK = 128


def quantize_q8_a(m):
    # check if m is 2D / matrix
    assert(len(m.shape) == 2)

    n_rows = m.shape[0]
    n_cols = m.shape[1]

    # check number of rows is divisible by QK
    assert(n_rows % QK == 0)

    # find scales
    scales = np.zeros([n_rows//QK, n_cols], np.float32)
    for i in range(n_rows//QK):
        for j in range(n_cols):
            # find max absolute value in sliced block
            scales[i][j] = np.max(np.abs(m[i:i+QK, j]))
    # we want to map qs to -128..127 range
    # convert to int8
    q = np.zeros([n_rows, n_cols], np.float32)
    for i in range(n_rows):
        for j in range(n_cols):
            q[i][j] = m[i][j] * (127.0/scales[i//QK][j])
    q = np.clip(q, -128, 127).astype(np.int8)

    return (q, scales)

def export_q8_a(res, f):
    (q, scales) = res
    n_rows = q.shape[0]
    n_cols = q.shape[1]

    # serialize to file
    # write header: magic 4 byte data type, n_rows, n_cols
    fout.write(b"q8_a")
    fout.write(struct.pack('ii', n_rows, n_cols))
    # scales
    scales = scales.flatten()
    fout.write(struct.pack(f'{len(scales)}f', *scales))
    # quantized values
    q = q.flatten()
    fout.write(struct.pack(f'{len(q)}b', *q))


def quantize_q8_b(m):
    # check if m is 2D / matrix
    assert(len(m.shape) == 2)

    n_rows = m.shape[0]
    n_cols = m.shape[1]

    # check number of rows is divisible by QK
    assert(n_cols % QK == 0)

    # find scales
    scales = np.zeros([n_rows, n_cols//QK], np.float32)
    means = np.zeros([n_rows, n_cols//QK], np.float32)
    for i in range(n_rows):
        for j in range(n_cols//QK):
            # find max absolute value in sliced block
            means[i][j] = np.mean(m[i, j*QK:(j+1)*QK])
            scales[i][j] = np.max(np.abs(m[i, j*QK:(j+1)*QK] - means[i][j]))
    # we want to map qs to -128..127 range
    # convert to int8
    q = np.zeros([n_rows, n_cols], np.float32)
    for i in range(n_rows):
        for j in range(n_cols):
            q[i][j] = (m[i][j] - means[i][j//QK]) * (127.0 / scales[i][j//QK])

    q = np.clip(q, -128, 127).astype(np.int8)
    return (q, scales, means)


def export_q8_b(res, f):
    (q, scales, means) = res
    n_rows = q.shape[0]
    n_cols = q.shape[1]

    # serialize to file
    # write header: magic 4 byte data type, n_rows, n_cols
    fout.write(b"q8_b")
    fout.write(struct.pack('ii', n_rows, n_cols))
    # scales
    scales = scales.flatten()
    fout.write(struct.pack(f'{len(scales)}f', *scales))
    means = means.flatten()
    fout.write(struct.pack(f'{len(means)}f', *means))
    # quantized values
    q = q.flatten()
    fout.write(struct.pack(f'{len(q)}b', *q))


def quantize_q4_a(m, f):

    def binz(vals):
        vals = sorted(vals)
        assert len(vals) % 16 == 0
        bins = np.zeros(16)
        bin_size = len(vals)//16
        for i in range(16):
            bins[i] = np.mean(vals[i*bin_size:(i+1)*bin_size])
        return bins

    def find_nearest(array, value):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
            return idx-1
        else:
            return idx

    # check if m is 2D / matrix
    assert(len(m.shape) == 2)

    n_rows = m.shape[0]
    n_cols = m.shape[1]

    # check number of rows is divisible by QK
    assert(n_cols % QK == 0)
    assert((n_cols//QK) % 2 == 0)
    # find bins
    bins = np.zeros([n_rows, n_cols//QK//2, 16], np.float32)
    for i in range(n_rows):
        for j in range(n_cols//QK//2):
            # find max absolute value in sliced block
            bins[i][j] = binz(m[i, j*QK*2:(j+1)*QK*2])
            # print("vals: ", m[i, j*QK*2:(j+1)*QK*2])
            # print("bins: ", bins[i][j])

    # convert to indexes
    q = np.zeros([n_rows, n_cols//2], np.uint8)
    for i in range(n_rows):
        for j in range(n_cols//2):
            idx0 = find_nearest(bins[i][j//QK], m[i][j*2])
            idx1 = find_nearest(bins[i][j//QK], m[i][j*2+1])

            #print("bins: ", bins[i][j])
            #print("val0: ", m[i][j*2], " idx0: ", idx0)
            #print("val1: ", m[i][j*2+1], " idx1: ", idx1)
            q[i][j] = idx0 << 4 | idx1
            #print("q: ", q[i][j])

    # serialize to file
    # write header: magic 4 byte data type, n_rows, n_cols
    fout.write(b"q4_a")
    fout.write(struct.pack('ii', n_rows, n_cols))
    # bins
    bins = bins.flatten()
    print(f"bins flatten: {len(bins)}")
    fout.write(struct.pack(f'{len(bins)}f', *bins))
    # indices
    q = q.flatten()
    print(f"q flatten: {len(q)}")
    fout.write(struct.pack(f'{len(q)}B', *q))
    print(f"cur: {fout.tell()}")

def test_q4():
    fout = open("test.q4", "wb")
    r = np.random.rand(768, 768)
    quantize_q4_a(r, fout)
    fout.close()

if len(sys.argv) != 3:
    print("quant.py <input model.bin file> <out model.q8 file>")
    sys.exit(1)

infile = sys.argv[1]
outfile = sys.argv[2]

pool = Pool(32)

f = open(infile, "rb")
fout = open(outfile, "wb")

header = f.read(7*4)
fout.write(header)

header = struct.unpack("iiiiiii", header)
dim = header[0]
hidden_dim = header[1]
n_layers = header[2]
n_heads = header[2]
n_vocab = header[5]
seq_len = header[6]

shared_weights = False
if n_vocab < 0:
    shared_weights = True
    n_vocab = -n_vocab

print(f"dim = {dim}")
print(f"layers = {n_layers}")
print(f"vocab = {n_vocab}")

# quantize embedding matrix
print("Quantizing token embedding")
w = np.fromfile(f, dtype=np.float32, count=dim*n_vocab).reshape((n_vocab, dim))
export_q8_b(quantize_q8_b(w), fout)

# skip rms attn weights
fout.write(f.read(dim*n_layers*4))


qs = []
for label in ["WQ", "WK", "WV", "WO"]:
    for i in range(n_layers):
        print(f"Quantizing {label} - layer {i}")
        w = np.fromfile(f, dtype=np.float32, count=dim*dim).reshape((dim, dim))
        qs.append(pool.apply_async(quantize_q8_a, (w,)))
for res in qs:
    export_q8_a(res.get(), fout)

# skip rms ffn weights
fout.write(f.read(dim*n_layers*4))

qs = []
for i in range(n_layers):
    print(f"Quantizing w1 - layer {i}")
    w = np.fromfile(f, dtype=np.float32, count=dim*hidden_dim).reshape((hidden_dim, dim))
    qs.append(pool.apply_async(quantize_q8_b, (w,)))
for res in qs:
    export_q8_b(res.get(), fout)

qs = []
for i in range(n_layers):
    print(f"Quantizing w2 - layer {i}")
    w = np.fromfile(f, dtype=np.float32, count=hidden_dim*dim).reshape((dim, hidden_dim))
    qs.append(pool.apply_async(quantize_q8_b, (w,)))
for res in qs:
    export_q8_b(res.get(), fout)

qs = []
for i in range(n_layers):
    print(f"Quantizing w3 - layer {i}")
    w = np.fromfile(f, dtype=np.float32, count=dim*hidden_dim).reshape((hidden_dim, dim))
    qs.append(pool.apply_async(quantize_q8_b, (w,)))
for res in qs:
    export_q8_b(res.get(), fout)


# skip rms final weights
fout.write(f.read(dim*4))

# skip rope/freq coeffs
head_size = dim // n_heads
fout.write(f.read(seq_len*head_size*4))

# quantize embedding matrix
if shared_weights:
    print("Quantizing wcls")
    w = np.fromfile(f, dtype=np.float32, count=dim*n_vocab).reshape((n_vocab, dim))
    export_q8_b(quantize_q8_b(w), fout)

# copy rest of file -
# by now nothing should be there..
rest = f.read()
print(f"left {len(rest)} bytes.")
fout.write(rest)
fout.close()

pool.close()
