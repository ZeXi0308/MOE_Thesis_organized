"""Offline tokenizer backend; no model/Transformers dependency required.

The fallback renderer is deliberately limited to the one-user-message shape
used here. It expands the exact pinned OLMoE template, not a general Jinja
interpreter. Every measured rendering and token hash is checked independently
against the existing fixed-revision source input by prepare_inputs.py.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path

OLMOE_TEMPLATE = "{{ bos_token }}{% for message in messages %}\n{% if message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] }}\n{% elif message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


class OfflineTokenizer:
    def __init__(self, folder):
        from tokenizers import Tokenizer
        folder = Path(folder)
        self.tokenizer = Tokenizer.from_file(str(folder/'tokenizer.json'))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.config = json.loads((folder/'tokenizer_config.json').read_text())
        self.receipt = dict(backend='tokenizers.Tokenizer',
                            version=importlib.metadata.version('tokenizers'),
                            renderer='restricted_exact_pinned_single_user_template_expansion',
                            jinja_executed=False, transformers_required=False,
                            template_sha256=hashlib.sha256(self.config.get('chat_template','').encode()).hexdigest())

    def encode(self, text, add_special_tokens=False):
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids, skip_special_tokens=True):
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def render_user(self, content):
        if (self.config.get('chat_template') != OLMOE_TEMPLATE
                or self.config.get('bos_token') != '<|endoftext|>'
                or self.config.get('eos_token') != '<|endoftext|>'):
            raise ValueError('Fallback only supports the exact pinned OLMoE one-user template')
        # Literal expansion of the selected user branch, with the whitespace
        # convention already fixed by the original 32 rendered source prompts.
        return '<|endoftext|><|user|>\n' + content + '\n<|assistant|>\nAnswer:'
