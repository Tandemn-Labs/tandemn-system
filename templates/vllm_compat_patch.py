# Compatibility patch for vLLM 0.10.0 with transformers 5.x and huggingface_hub
import transformers.tokenization_utils_base as tub

# Patch 1: Add missing all_special_tokens_extended (removed in transformers 4.47+)
if not hasattr(tub.PreTrainedTokenizerBase, 'all_special_tokens_extended'):
    def _all_special_tokens_extended(self):
        """Returns all special tokens including added tokens."""
        all_toks = list(self.all_special_tokens)
        for tok in self.added_tokens_encoder.keys():
            if tok not in all_toks:
                all_toks.append(tok)
        return all_toks
    tub.PreTrainedTokenizerBase.all_special_tokens_extended = property(_all_special_tokens_extended)
    print("Patched transformers: added all_special_tokens_extended")

# Patch 2: Fix vLLM DisabledTqdm duplicate 'disable' kwarg issue
# This is fixed by pinning huggingface_hub in setup. This patch is a backup.
def _apply_tqdm_patch():
    """Patch tqdm to handle duplicate disable kwarg gracefully."""
    try:
        import tqdm.std
        _orig_init = tqdm.std.tqdm.__init__

        def _fixed_init(self, *args, **kwargs):
            # If disable appears in both args position and kwargs, remove from kwargs
            return _orig_init(self, *args, **kwargs)

        # Don't patch - rely on huggingface_hub version pin instead
        # tqdm.std.tqdm.__init__ = _fixed_init
    except Exception:
        pass

_apply_tqdm_patch()
