import argparse
import gc
import os
import re
import sys
import abc
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaForCausalLM,
    T5Tokenizer,
)
import os
import logging
import json
import numpy


from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory

# Load Gorilla Model from HF
def load_model(
        model_path: str,
        device: str,
        num_gpus: int,
        max_gpu_memory: str = None,
        load_8bit: bool = False,
        cpu_offloading: bool = False,
    ):
 
    if device == "cpu":
        kwargs = {"torch_dtype": torch.float32}
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus != 1:
            kwargs["device_map"] = "auto"
            if max_gpu_memory is None:
                kwargs[
                    "device_map"
                ] = "sequential"  # This is important for not the same VRAM sizes
                available_gpu_memory = get_gpu_memory(num_gpus)
                kwargs["max_memory"] = {
                    i: str(int(available_gpu_memory[i] * 0.85)) + "GiB"
                    for i in range(num_gpus)
                }
            else:
                kwargs["max_memory"] = {i: max_gpu_memory for i in range(num_gpus)}
    else:
        raise ValueError(f"Invalid device: {device}")

    if cpu_offloading:
        # raises an error on incompatible platforms
        from transformers import BitsAndBytesConfig

        if "max_memory" in kwargs:
            kwargs["max_memory"]["cpu"] = (
                str(math.floor(psutil.virtual_memory().available / 2**20)) + "Mib"
            )
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit_fp32_cpu_offload=cpu_offloading
        )
        kwargs["load_in_8bit"] = load_8bit
    elif load_8bit:
        if num_gpus != 1:
            warnings.warn(
                "8-bit quantization is not supported for multi-gpu inference."
            )
        else:
            return load_compress_model(
                model_path=model_path, device=device, torch_dtype=kwargs["torch_dtype"]
            )
  
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = 11
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        **kwargs,
    )

    return model, tokenizer

@torch.inference_mode()
def get_response(prompt, model, tokenizer, device):
    input_ids = tokenizer([prompt]).input_ids
    output_ids = model.generate(
        torch.as_tensor(input_ids).to(device),
        do_sample=True,
        temperature=0.7,
        max_new_tokens=1024,
        pad_token_id=tokenizer.eos_token_id
    )
    output_ids = output_ids[0][len(input_ids[0]) :]
    outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    yield {"text": outputs}

    # clean
    gc.collect()
    torch.cuda.empty_cache()

def stream_output(output_stream):
        pre = 0
        for outputs in output_stream:
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                print(" ".join(output_text[pre:now]), end=" ", flush=True)
                pre = now
        print(" ".join(output_text[pre:]), flush=True)
        return " ".join(output_text)

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    global tokenizer
    global device

    logging.info("model 1: request received")
    # Model
    device = "cuda"
    # export AZUREML_MODEL_DIR=/home/azureuser/cloudfiles/code/gorilla-falcon-7b-hf-v0
    model_path = os.getenv("AZUREML_MODEL_DIR")
    
    model, tokenizer = load_model(
        model_path, device, 1, None, False, False
    )
    
    model.to(device)
    logging.info("Init complete")
    return model, tokenizer



def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    logging.info("model 1: request received")
    prompt = json.loads(raw_data)["data"]
    output_stream = get_response(prompt, model, tokenizer, device)
    outputs = stream_output(output_stream)
    logging.info("Request processed")
    return outputs

def main():
    prompt = "I need to tranlate from english to spanish"
    output_stream = get_response(prompt, model, tokenizer, device)
    outputs = stream_output(output_stream)
    print(outputs)

# init()
# main()