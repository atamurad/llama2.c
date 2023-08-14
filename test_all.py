"""
Run simply with
$ pytest
"""
import os
import pytest # pip install pytest
import requests
import subprocess


import torch
from model import ModelArgs, Transformer
from tokenizer import Tokenizer

import sentencepiece as spm
from ctypes import c_char_p, c_int, c_uint, c_float, cdll, byref, POINTER


# -----------------------------------------------------------------------------
# test utilities

test_ckpt_dir = "test"

def download_file(url, filename):
    print(f"Downloading {url} to {filename}")
    response = requests.get(url, stream=True)
    response.raise_for_status() # Raise an HTTPError on bad status code
    with open(filename, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

def attempt_download_files():
    os.makedirs(test_ckpt_dir, exist_ok=True)
    root_url = "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K"
    need = ["stories260K.bin", "stories260K.pt", "tok512.bin", "tok512.model"]
    for file in need:
        url = os.path.join(root_url, file)
        filename = os.path.join(test_ckpt_dir, file)
        if not os.path.exists(filename):
            download_file(url, filename)

def load_runc():
    runc = cdll.LoadLibrary("./run.so")
    runc.load_tokenizer.argtype = [c_char_p, c_int, POINTER(c_char_p), POINTER(c_float), POINTER(c_int)]
    runc.bpe_encode.argtype = [c_char_p, POINTER(c_char_p), POINTER(c_float), c_int, c_uint, POINTER(c_int), POINTER(c_int)]
    return runc


expected_stdout = b'Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but it was too high.\nLily\'s mom said, "Lily, let\'s go to the park." Lily was sad and didn\'t know what to do. She said, "I want to play with your ball, but I can\'t find it."\nLily was sad and didn\'t know what to do. She said, "I\'m sorry, Lily. I didn\'t know what to do."\nLily didn\'t want to help her mom, so she'

# -----------------------------------------------------------------------------
# actual tests

def test_runc():
    """ Forwards a model against a known-good desired outcome in run.c for 200 steps"""
    attempt_download_files()

    model_path = os.path.join(test_ckpt_dir, "stories260K.bin")
    tokenizer_path = os.path.join(test_ckpt_dir, "tok512.bin")
    command = ["./run", model_path, "-z", tokenizer_path, "-t", "0.0", "-n", "200"]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    stdout, stderr = proc.communicate()

    # strip the very last \n that is added by run.c for aesthetic reasons
    stdout = stdout[:-1]
    assert stdout == expected_stdout

def test_python():
    """ Forwards a model against a known-good desired outcome in sample.py for 200 steps"""
    attempt_download_files()

    device = "cpu" # stories260K is small enough to just breeze through it on CPU
    checkpoint = os.path.join(test_ckpt_dir, "stories260K.pt")
    checkpoint_dict = torch.load(checkpoint, map_location=device)
    gptconf = ModelArgs(**checkpoint_dict['model_args'])
    model = Transformer(gptconf)
    state_dict = checkpoint_dict['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    x = torch.tensor([[1]], dtype=torch.long, device=device) # 1 is BOS
    with torch.inference_mode():
        y = model.generate(x, max_new_tokens=200, temperature=0.0)
    pt_tokens = y[0].tolist()

    tokenizer_model = os.path.join(test_ckpt_dir, "tok512.model")
    enc = Tokenizer(tokenizer_model=tokenizer_model)
    text = enc.decode(pt_tokens)
    text = text.encode('ascii') # turn into bytes

    assert text == expected_stdout


# -----------------------------------------------------------------------------
# bpe tokenizer tests

test_prompts = ["hello world!",
                "abcdefgh",
                "0123456789",
                "Lets try ö & 株式会社",
                "𒎗𓐍",
                "[INST]\n<<<SYS>>>test<<</SYS>>>\nhello[/INST]\n"]


def test_load_tokenizer():
    """Loads tok512.bin with run.c, tok512.model with sentencepiece and compare all tokens and scores"""
    attempt_download_files()

    tokenizer_path = os.path.join(test_ckpt_dir, "tok512.bin")
    tokenizer_model_path = os.path.join(test_ckpt_dir, "tok512.model")

    runc = load_runc()

    # load tokenizer with run.c
    vocab_size = 512
    vocab = (c_char_p * vocab_size)()
    vocab_scores = (c_float * vocab_size)()
    max_token_len = c_int()

    runc.load_tokenizer(tokenizer_path.encode(), vocab_size, vocab, vocab_scores, byref(max_token_len))

    assert max_token_len.value == 7

    # load tokenizer with sentencepiece
    s = spm.SentencePieceProcessor(model_file=tokenizer_model_path)
    assert vocab_size == s.vocab_size()

    bos_id = s.bos_id()
    eos_id = s.eos_id()

    # compare all tokens
    for i in range(vocab_size):
        # tokenizer.py export does some light transofrmation
        t = s.id_to_piece(i)
        if i == bos_id:
            t = '\n<s>\n'
        elif i == eos_id:
            t = '\n</s>\n'
        t = t.replace('▁', ' ')
        b = t.encode('utf-8')

        assert vocab[i] == b
        assert vocab_scores[i] == s.get_score(i)


def test_bpe_encode():
    """Test run.c bpe_encode() output with expected spm.encode() output for given test prompts"""
    attempt_download_files()

    tokenizer_path = os.path.join(test_ckpt_dir, "tok512.bin")
    tokenizer_model_path = os.path.join(test_ckpt_dir, "tok512.model")

    runc = load_runc()
    # load tokenizer with run.c
    vocab_size = 512
    vocab = (c_char_p * vocab_size)()
    vocab_scores = (c_float * vocab_size)()
    max_token_len = c_int()
    runc.load_tokenizer(tokenizer_path.encode(), vocab_size, vocab, vocab_scores, byref(max_token_len))

    # load tokenizer with sentencepiece
    s = spm.SentencePieceProcessor(model_file=tokenizer_model_path)

    tokens = (c_int * 1024)()
    n_tokens = c_int()

    for prompt in test_prompts:
        print(f"Testing bpe_encode with prompt '{prompt}'")
        # encode with run.c
        runc.bpe_encode(prompt.encode("utf8"), vocab, vocab_scores, vocab_size, max_token_len, tokens, byref(n_tokens))
        # encode with spm
        expected = s.encode(prompt)

        print(f"Expected: {expected}")

        assert len(expected) == n_tokens.value

        for i in range(n_tokens.value):
            assert tokens[i] == expected[i]
