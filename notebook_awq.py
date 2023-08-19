import numpy as np

# int8 paper: some columns (~1%) require more precision, rest can be 4 bit
# split weight matrix into 2: 4 bit part and 16 bit part

# awq paper:
# * mixed precision matrix / splitting matrix is not hardware friendly
# * can't tell which columns from weights of matrix alone, need to look at activations/calibration
# * int4 quantization provides more precision to max/extreme values
# * input vector and weight matrix can be scaled to emphasize columns that need more precision
# * all matmul weights are int4
# * scaling factor for input vector can be fused to previous layer / op

# seaching scaling factors outside scope of this code/notebook


def quantize_q4(m):
    # find max
    scale = np.max(np.abs(m))

    n_rows = m.shape[0]
    n_cols = m.shape[1]
    # we want to map qs to -128..127 range
    # convert to int8
    q = np.zeros([n_rows, n_cols], np.float32)
    for i in range(n_rows):
        for j in range(n_cols):
            q[i][j] = m[i][j] * (15.0/scale)
    q = np.clip(q, -16, 15).astype(np.int8)

    return (scale, q)


def dequantize_q4(scale, q):
    n_rows = q.shape[0]
    n_cols = q.shape[1]
    m = np.zeros([n_rows, n_cols], np.float32)
    for i in range(n_rows):
        for j in range(n_cols):
            m[i][j] = q[i][j] * (scale/15.0)
    return m


def awq_matmul(x, m):
    # find s
    s = np.array([1, 100, 1])

    # x` = scale x*s
    x_p = x / s
    # m' = apply inverse scale to m
    m_p = (s * m.T).T

    # m_p - simulate quantization loss
    (q4scale, qm) = quantize_q4(m_p)
    m_p = dequantize_q4(q4scale, qm)

    print("Dequantized AWQ")
    print(m_p)

    # -> dequantize and matrix multiply
    out = x_p.dot(m_p)
    return out


def compare_awq_loss():
    x = np.array([2, 100.0, 1])

    m = np.array([
        [10,   1,  0,      0],
        [0.5,  0.9,  0.5,  0.5],
        [0,    2,  20,    30],
    ])

    print("reference matrix:")
    print(m)

    print("quantize q4:")
    q4scale, qm = quantize_q4(m)
    print(q4scale)
    print(qm)

    print("dequantize q4:")
    qm = dequantize_q4(q4scale, qm)
    print(q4scale)
    print(qm)

    print("\n\nActivation")
    print("-----------------------\n")
    print("reference activation:")
    out_ref = x.dot(m)
    print(out_ref)

    print("q4 activation:")
    out_q4 = x.dot(qm)
    print(out_q4)

    print("Q4 Error (MSE):")
    error = out_ref - out_q4
    print(error)
    print((error**2).mean())

    print("\n\nAWQ Activation")
    print("-----------------------\n")

    out_awq = awq_matmul(x, m)

    print("q4 activation:")
    print(out_awq)

    print("AWQ error (MSE):")
    error = out_ref - out_awq
    print(error)
    print((error**2).mean())


compare_awq_loss()
