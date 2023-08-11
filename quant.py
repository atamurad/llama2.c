import numpy as np
import struct
import os

QK = 64


def quantize(m, f):
    # check m is 2D matrix
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
    scales /= 128.0
    # convert to int8
    q = np.zeros([n_rows, n_cols], np.int8)
    for i in range(n_rows):
        for j in range(n_cols):
            q[i][j] = m[i][j] / scales[i//QK][j]

    # serialize to file
    # write header: magic 4 byte data type, n_rows, n_cols
    fout.write(b"q8_0")
    fout.write(struct.pack('ii', n_rows, n_cols))
    # scales
    scales = scales.flatten()
    fout.write(struct.pack(f'{len(scales)}f', *scales))
    # quantized values
    q = q.flatten()
    fout.write(struct.pack(f'{len(q)}b', *q))


def test_quantize():
    fout = open("test.q8", "wb")
    m = np.random.rand(512, 512)
    quantize(m, fout)
    fout.close()


f = open("out110m/model110m.bin", "rb")
fout = open("model110.q8", "wb")

header = f.read(7*4)
fout.write(header)

header = struct.unpack("iiiiiii", header)
dim = header[0]
n_layers = header[2]
n_vocab = header[5]

print(f"dim = {dim}")
print(f"layers = {n_layers}")
print(f"vocab = {n_vocab}")

# skip embedding matrix
fout.write(f.read(dim*n_vocab*4))
# skip rms weights
fout.write(f.read(dim*n_layers*4))

for label in ["WQ", "WK", "WV", "WO"]:
    for i in range(n_layers):
        print(f"Quantizing {label} - layer {i}")
        w = np.fromfile(f, dtype=np.float32, count=dim*dim).reshape((dim, dim))
        quantize(w, fout)

# copy rest of file
rest = f.read()
print(f"rest of file {len(rest)}")
fout.write(rest)
fout.close()

