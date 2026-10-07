import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["LD_LIBRARY_PATH"] = (
    os.environ.get("LD_LIBRARY_PATH", "") + ":/usr/local/lib/libctranslate2.so.4"
)


from typing import List, Union, Tuple
import ctranslate2
import sentencepiece as spm


sp = spm.SentencePieceProcessor()
sp.Load("/media/bering/8TB1/teryn/enko_training_toolkit/data/enko.model")
# sp.Load("/home/bering/teryn/nsf_sbir/nllb.model")

translator = ctranslate2.Translator(
    "/media/bering/8TB1/teryn/enko_training_toolkit/ct2_model",
    device="cpu",
    inter_threads=1,
    intra_threads=1,
)

from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=12)

import time

s = time.time()
translator.translate_batch([sp.EncodeAsPieces("한국어")])
print(time.time() - s)


s = time.time()
futures = [
    executor.submit(translator.translate_batch, [sp.EncodeAsPieces("한국어")])
    for _ in range(12)
]
[f.result() for f in futures]
print(time.time() - s)
