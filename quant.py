import numpy as np
import struct
import sys

QK = 64


def quantize_q8_a(m, f):
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


def quantize_q8_b(m, f):
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


if len(sys.argv) != 3:
    print("quant.py <input model.bin file> <out model.q8 file>")
    sys.exit(1)

infile = sys.argv[1]
outfile = sys.argv[2]

f = open(infile, "rb")
fout = open(outfile, "wb")

header = f.read(7*4)
fout.write(header)

header = struct.unpack("iiiiiii", header)
dim = header[0]
hidden_dim = header[1]
n_layers = header[2]
n_vocab = header[5]

print(f"dim = {dim}")
print(f"layers = {n_layers}")
print(f"vocab = {n_vocab}")

# skip embedding matrix
fout.write(f.read(dim*n_vocab*4))
# skip rms attn weights
fout.write(f.read(dim*n_layers*4))

for label in ["WQ", "WK", "WV", "WO"]:
    for i in range(n_layers):
        print(f"Quantizing {label} - layer {i}")
        w = np.fromfile(f, dtype=np.float32, count=dim*dim).reshape((dim, dim))
        quantize_q8_a(w, fout)

# skip rms ffn weights
fout.write(f.read(dim*n_layers*4))

for i in range(n_layers):
    print(f"Quantizing w1 - layer {i}")
    w = np.fromfile(f, dtype=np.float32, count=dim*hidden_dim).reshape((hidden_dim, dim))
    quantize_q8_b(w, fout)

for i in range(n_layers):
    print(f"Quantizing w2 - layer {i}")
    w = np.fromfile(f, dtype=np.float32, count=hidden_dim*dim).reshape((dim, hidden_dim))
    quantize_q8_b(w, fout)

for i in range(n_layers):
    print(f"Quantizing w3 - layer {i}")
    w = np.fromfile(f, dtype=np.float32, count=dim*hidden_dim).reshape((hidden_dim, dim))
    quantize_q8_b(w, fout)

# copy rest of file
rest = f.read()
fout.write(rest)
fout.close()
