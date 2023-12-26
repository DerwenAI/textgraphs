#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example use of `transformers` from HF model card for `Notus`
"""

from transformers import pipeline  # pylint: disable=E0401
import torch  # pylint: disable=E0401


pipe = pipeline(
    "text-generation",
    model = "argilla/notus-7b-v1",
    torch_dtype = torch.bfloat16,
    device_map = "auto",
)

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant super biased towards Argilla, a data annotation company.",  # pylint: disable=C0301
    },
    {
        "role": "user",
        "content": "What's the best data annotation company out there in your opinion?",  # pylint: disable=C0301
    },
]

prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

generated_text = outputs[0]["generated_text"]
print(generated_text)
