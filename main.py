import logging
import os
import time
import requests
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
import gfmrag
from gfmrag.llms import BaseLanguageModel
from gfmrag.prompt_builder import QAPromptBuilder

from gfmrag import GFMRetriever

logger = logging.getLogger(__name__)

model_name = "meta-llama/Llama-2-7b-chat-hf"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
cache_dir = "../cache_dir"
llm_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    torch_dtype=torch.float16
).to(device)

max_new_tokens = {
    "Qwen/Qwen2.5-7B-Instruct": 2048,
    "meta-llama/Llama-2-7b-chat-hf": 2048
}

LMUNIT_TEST = "Is the response correct? Groundtruth: {groundtruth}"

def generate(messages, context=""): # Comes from GNN-RAG generate_dataset_from_hf.py file
    llm_prompt = [tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )]
    inputs = tokenizer(llm_prompt, return_tensors="pt").to(device)
    inputs_len = inputs.input_ids.size(-1)
    outputs = llm_model.generate(**inputs, max_new_tokens=max_new_tokens[model_name])
    response = tokenizer.decode(outputs[0][inputs_len:], skip_special_tokens=True)
    return response

def evaluate_llm(messages, groundtruth, throttle_time=1): # Comes from models/ReaRev/rearev.py file
    score = 0
    start_time = time.time()
    try:
        prediction = generate(messages).strip()
    except:
        print("Failed on generate sentence")
        print(f"Curr input: {messages}")
    unit_test = LMUNIT_TEST.format(groundtruth=groundtruth)
    url = "https://api.contextual.ai/v1/lmunit"
    lm_unit_api_key = os.getenv("LM_UNIT_API_KEY")
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {lm_unit_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": messages[-1]["content"],
        "response": prediction,
        "unit_test": unit_test
    }
    print(payload)
    time_elapsed = time.time() - start_time
    if time_elapsed < throttle_time:
        time.sleep(throttle_time - time_elapsed)
    response = requests.post(url, json=payload, headers=headers)
    if response.ok:
        score = response.json().get("score")
    else:
        print(f"Response not ok: {response.json()}")         
    return score

@hydra.main(
    config_path="config", config_name="stage3_qa_ircot_inference", version_base=None
)
def main(cfg: DictConfig) -> None:
    output_dir = HydraConfig.get().runtime.output_dir
    logger.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Output directory: {output_dir}")

    current_query = "Who is the president of France?"
    groundtruth = "The president of France is Emmanuel Macron"
    retriever = GFMRetriever.from_config(cfg)
    docs = retriever.retrieve(current_query, top_k=1)

    qa_prompt_builder = QAPromptBuilder(cfg.qa_prompt)
    messages = qa_prompt_builder.build_input_prompt(current_query, docs)
    print(evaluate_llm(messages, groundtruth))

main()
