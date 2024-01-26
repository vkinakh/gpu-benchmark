import time

import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import setup_logger


# 10*10 Prompts. Source: https://www.penguin.co.uk/articles/2022/04/best-first-lines-in-books
prompts_all = [
    "The King is dead. Long live the Queen.",
    "Once there were four children whose names were Peter, Susan, Edmund, and Lucy.",
    "The story so far: in the beginning, the universe was created.",
    "It was a bright cold day in April, and the clocks were striking thirteen.",
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
    "The sweat wis lashing oafay Sick Boy; he wis trembling.",
    "124 was spiteful. Full of Baby's venom.",
    "As Gregor Samsa awoke one morning from uneasy dreams he found himself transformed in his bed into a gigantic insect.",
    "I write this sitting in the kitchen sink.",
    "We were somewhere around Barstow on the edge of the desert when the drugs began to take hold.",
] * 10

LLM_MODEL_NAMES = [
    "meta-llama/Llama-2-7b-chat-hf",
    "sgugger/sharded-gpt-j-6B",
]


def llm_inference(
    model_name: str,
    log_filename: str,
) -> None:
    if model_name not in LLM_MODEL_NAMES:
        raise ValueError("Invalid model name.")

    logger = setup_logger(log_filename)

    tokenizer_name_dict = {
        "meta-llama/Llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
        "sgugger/sharded-gpt-j-6B": "EleutherAI/gpt-j-6B",
    }
    tokenizer_name = tokenizer_name_dict[model_name]

    accelerator = Accelerator()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": accelerator.process_index},
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # sync GPUs and start the timer
    accelerator.wait_for_everyone()
    start = time.time()

    # divide the prompt list onto the available GPUs
    with accelerator.split_between_processes(prompts_all) as prompts:
        # store output of generations in dict
        results = dict(outputs=[], num_tokens=0)

        # have each GPU do inference, prompt by prompt
        for prompt in prompts:
            prompt_tokenized = tokenizer(prompt, return_tensors="pt").to("cuda")
            output_tokenized = model.generate(**prompt_tokenized, max_new_tokens=100)[0]

            # remove prompt from output
            output_tokenized = output_tokenized[len(prompt_tokenized["input_ids"][0]) :]

            # store outputs and number of tokens in result{}
            results["outputs"].append(tokenizer.decode(output_tokenized))
            results["num_tokens"] += len(output_tokenized)

        results = [
            results
        ]  # transform to list, otherwise gather_object() will not collect correctly

    # collect results from all the GPUs
    results_gathered = gather_object(results)

    if accelerator.is_main_process:
        timediff = time.time() - start
        num_tokens = sum([r["num_tokens"] for r in results_gathered])

        logger.info(
            f"Model: {model_name}, "
            f"tokenizer: {tokenizer_name}, "
            f"tokens/sec: {num_tokens//timediff}, "
            f"time {timediff}, "
            f"total tokens {num_tokens}, "
            f"total prompts {len(prompts_all)}"
        )
