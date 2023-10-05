import sys
import time
from pathlib import Path
from typing import Literal, Optional, List

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
from torch.nn.utils.rnn import pad_sequence

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import GPT, Config, Tokenizer
from lit_gpt.model import Block
from lit_gpt.utils import check_valid_checkpoint_dir, get_default_supported_precision, load_checkpoint, quantization


@torch.inference_mode()
def generate(
    model: GPT,
    x: torch.Tensor,
    mask: torch.Tensor,
    max_new_tokens: int,
    min_T: int,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> List[torch.Tensor]:
    device, dtype = x.device, x.dtype
    input_pos = torch.arange(min_T, device=device)
    done = torch.zeros(len(x), dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        # forward
        logits = model(x.index_select(1, input_pos), input_pos)
        logits = logits[:, -1]

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(logits < v[[-1]], -float("Inf"), logits)

        if temperature > 0:
            probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
            next = torch.multinomial(probs, num_samples=1).to(dtype=dtype)
        else:
            next = torch.argmax(logits, dim=-1)

        # advance
        input_pos = input_pos[-1:] + 1

        # concatenate the new generation
        x[:, input_pos] = torch.where(mask[:, input_pos], x[:, input_pos], next)

        # if <eos> token is triggered, return the output (stop generation)
        done |= next == eos_id
        if all(done):
            return x[:, :input_pos].unbind()  # include the EOS token

    return x.unbind()


def main(
    prompts: List[str] = ["Hello, my name is", "Hello, my name is"],
    *,
    num_samples: int = 1,
    max_new_tokens: int = 50,
    top_k: Optional[int] = 200,
    temperature: float = 0.8,
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8", "gptq.int4"]] = None,
    strategy: str = "auto",
    devices: int = 1,
    precision: Optional[str] = None,
) -> None:
    """Generates text samples based on a pre-trained model and tokenizer.

    Args:
        prompt: The prompt string to use for generating the samples.
        num_samples: The number of text samples to generate.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        checkpoint_dir: The checkpoint directory to load.
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            - bnb.int8: 8-bit quantization from bitsandbytes
            - gptq.int4: 4-bit quantization from GPTQ
            for more details, see https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/quantize.md
        strategy: Indicates the Fabric strategy setting to use.
        devices: How many devices to use.
        precision: Indicates the Fabric precision setting to use.
    """
    precision = precision or get_default_supported_precision(training=False)

    if strategy == "fsdp":
        strategy = FSDPStrategy(auto_wrap_policy={Block}, cpu_offload=False)
    fabric = L.Fabric(devices=devices, precision=precision, strategy=strategy)
    fabric.launch()

    check_valid_checkpoint_dir(checkpoint_dir)

    config = Config.from_json(checkpoint_dir / "lit_config.json")

    if quantize is not None and devices > 1:
        raise NotImplementedError
    if quantize == "gptq.int4":
        model_file = "lit_model_gptq.4bit.pth"
        if not (checkpoint_dir / model_file).is_file():
            raise ValueError("Please run `python quantize/gptq.py` first")
    else:
        model_file = "lit_model.pth"
    checkpoint_path = checkpoint_dir / model_file

    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True), quantization(quantize):
        model = GPT(config)
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    t0 = time.perf_counter()

    if quantize:
        # for quantization, need to load before moving to device
        load_checkpoint(fabric, model, checkpoint_path, strict=(quantize is None))

    model.eval()
    model = fabric.setup_module(model)

    if not quantize:
        load_checkpoint(fabric, model, checkpoint_path, strict=(quantize is None))

    fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    tokenizer = Tokenizer(checkpoint_dir)

    tokens = [tokenizer.encode(p, device=fabric.device) for p in prompts]
    max_T = max(len(e) for e in tokens)
    min_T = min(len(e) for e in tokens)
    assert max_new_tokens > 0
    max_returned_tokens = max_T + max_new_tokens
    pad_id = 1234  # FIXME
    padded_tokens = pad_sequence(tokens, batch_first=True, padding_value=pad_id)
    x = torch.full((len(tokens), max_new_tokens), pad_id, dtype=padded_tokens.dtype, device=fabric.device)
    x = torch.cat((padded_tokens, x), dim=1)
    mask = x != pad_id

    with fabric.init_tensor():
        # set the max_seq_length to limit the memory usage to what we need
        model.max_seq_length = max_returned_tokens

    L.seed_everything(1234)
    for i in range(num_samples):
        with fabric.init_tensor():
            # enable the kv cache
            model.set_kv_cache(batch_size=len(prompts))
        if i != 0:
            # reset the input tensor
            x[mask] = pad_id

        t0 = time.perf_counter()
        ys = generate(model, x, mask, max_new_tokens, min_T, temperature=temperature, top_k=top_k)
        t = time.perf_counter() - t0

        for y in ys:
            fabric.print(tokenizer.decode(y))
        tokens_generated = max(len(y) - len(t) for y, t in zip(ys, tokens))
        fabric.print(
            f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr
        )
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(main)
