from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._sample_actions = nnx_utils.module_jit(
            model.sample_actions,
            static_argnames=("temperature", "n_action_samples")
        )

        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    def _infer_maybe_multi(self, obs: dict) -> dict:
        # Copied from https://github.com/vla-safe/openpi/commit/109b414bebf2188076ef0ecff989115fba20ddfc#diff-2f8396caaba5c7509e3fcc67a2ec0b25646db5a993e2120127daa12c38527df2
        # no support for torch

        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)

        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        self._rng, sample_rng = jax.random.split(self._rng)

        sample_action_outputs = self._sample_actions(
            sample_rng,
            _model.Observation.from_dict(inputs),
            **self._sample_kwargs
        )

        if isinstance(sample_action_outputs, tuple):
            if "Pi0.sample_actions" in self._sample_actions.__repr__():
                output_tokens, aux_outputs = sample_action_outputs
                output_tokens_truncated = output_tokens
            elif "Pi0FAST.sample_actions" in self._sample_actions.__repr__():
                output_tokens, aux_outputs = sample_action_outputs
                # process the output and cut off the unused positions
                step = aux_outputs["decode_step"].max()
                output_tokens_truncated = output_tokens[:, :step]
                aux_outputs['encoded'] = aux_outputs['encoded'][:, :step]
                aux_outputs['logits'] = aux_outputs['logits'][:, :step]
                aux_outputs['pre_logits'] = aux_outputs['pre_logits'][:, :step]
            else:
                raise ValueError(f"Unknown model type: {self._sample_actions.__repr__()}")
        else:
            output_tokens = sample_action_outputs
            output_tokens_truncated = output_tokens
            aux_outputs = {}
        
        batch_size = output_tokens.shape[0]
        outputs = {
            "state": inputs["state"].repeat(batch_size, 0),
            "actions": output_tokens
        }

        outputs_transformed = []
        for i in range(batch_size):
            outputs_transformed.append(
                self._output_transform(
                    jax.tree.map(lambda x: np.asarray(x[i, ...]), outputs)
                )
            )
        outputs_transformed = {
            k: jnp.stack([v[k] for v in outputs_transformed], axis=0)
            for k in outputs_transformed[0].keys()
        }

        outputs = {
            "state": inputs["state"].repeat(batch_size, 0),
            "raw_actions": output_tokens_truncated,
            "actions": outputs_transformed["actions"]
        }
        outputs.update(aux_outputs)

        return outputs
    
    @override 
    def infer(self, obs: dict) -> dict:
        outputs = self._infer_maybe_multi(obs)

        def _unbatch_leaf(x):
            x = np.asarray(x)
            
            if x.ndim == 0:
                return x
            
            return x[0]
        
        return jax.tree_util.tree_map(_unbatch_leaf, outputs)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results_multi = self._policy._infer_maybe_multi(obs)

        results = {}
        for k in results_multi:
            if k in ['actions', 'decode_step', "raw_actions"]:
                results[k] = np.asarray(results_multi[k])
            else:
                results[k] = np.asarray(results_multi[k][0])

        meta_to_save = {}
        for k, v in obs.items():
            if not "image" in k:
                meta_to_save[k] = v
        meta_to_save.update(results)

        # handle pi0 fast outputs
        if "logits" in meta_to_save:
            logits, start_index, end_index = self._process_pi0fast_logits(meta_to_save["logits"])
            meta_to_save["logits"] = logits
            meta_to_save["action_start_index_in_vocab"] = start_index
            meta_to_save["action_end_index_in_vocab"] = end_index

        # if multiple actions are sampled, remove the intermediate outputs
        if results_multi["actions"].shape[0] > 1:
            if "encoded" in meta_to_save: del meta_to_save['encoded']
            if "logits" in meta_to_save: del meta_to_save['logits']
            if "pre_logits" in meta_to_save: del meta_to_save['pre_logits']

        self._record_step += 1

        # return results (control results + embedding data)
        for k, v in meta_to_save.items():
            if isinstance(v, (np.ndarray, jnp.ndarray)):
                if v.dtype == jnp.bfloat16:
                    meta_to_save[k] = v.astype(jnp.float32)
        
        trimmed_results = {
            "state": np.asarray(results_multi["state"][0]),
            "actions": np.asarray(results_multi["actions"][0])
        }
        trimmed_results.update(meta_to_save)

        print("CALLED")
        print(trimmed_results)

        return trimmed_results

    def _process_pi0fast_logits(self, logits):
            '''
            About the meaning of different token indices in vocabulary
            
            In the normal case:
            The first three output tokens should be [4022, 235292, 235248], decoded to "Actions: ". 
            The last two output token should be [235371, 1], decoded to "|" and "<eos>".
            The rest of the predicted tokens will be decoded into actual action tokens (looks like <loc0594>). 

            The underlying tokenizer has two stages
            - self._output_transform.transforms[0].tokenizer._paligemma_tokenizer 
            converts the predicted token id to the corresponding token string. Vocab size is
            self._paligemma_tokenizer.vocab_size() (text_vocab_size)
            - self._output_transform.transforms[0].tokenizer._fast_tokenizer
            converts the action tokens to the action trajectories. Vocab size is 
            self._output_transform.transforms[0].tokenizer._fast_tokenizer.vocab_size (action_vocab_size)
            
            
            According to FASTTokenizer._act_tokens_to_paligemma_tokens() function, the action tokens are 
            from text_vocab_size - 1 - self._fast_skip_tokens - action_vocab_size + 1,
                corresponding to action_vocab_size - 1, inclusive
            to text_vocab_size - 1 - self._fast_skip_tokens 
                corresponding to 0, inclusive
                
            So the actual slicing index should be 
            [
                text_vocab_size - self._fast_skip_tokens - action_vocab_size:
                text_vocab_size - self._fast_skip_tokens
            ]
            '''
            
            text_tokenizer = self._policy._output_transform.transforms[0].tokenizer._paligemma_tokenizer
            action_tokenizer = self._policy._output_transform.transforms[0].tokenizer._fast_tokenizer
            text_vocab_size = text_tokenizer.vocab_size()
            action_vocab_size = action_tokenizer.vocab_size
            fast_skip_tokens = self._policy._output_transform.transforms[0].tokenizer._fast_skip_tokens
            start_index = text_vocab_size - fast_skip_tokens - action_vocab_size
            end_index = text_vocab_size - fast_skip_tokens
            logits = logits[:, start_index:end_index]
            
            return logits, start_index, end_index