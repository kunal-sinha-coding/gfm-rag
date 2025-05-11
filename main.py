import logging
import os
import time
import json
import requests
import torch
import numpy as np
import networkx as nx
from tqdm import tqdm

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
from gfmrag.kg_construction.utils import KG_DELIMITER

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
PROCESSED_FOLDER = os.path.join("data", "hotpotqa_test", "processed")
SRC_FOLDER = os.path.join("..", "GNN-RAG", "gnn", "data", "finqa-debug")
ENTITIES_FILE = os.path.join(SRC_FOLDER, "entities.txt")
RELATIONS_FILE = os.path.join(SRC_FOLDER, "relations.txt")
ENT2ID_FILE = os.path.join(PROCESSED_FOLDER, "stage2", "ent2id.json")
REL2ID_FILE = os.path.join(PROCESSED_FOLDER, "stage2", "rel2id.json")
KG_FILE = os.path.join(PROCESSED_FOLDER, "stage1", "kg.txt")

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
    time_elapsed = time.time() - start_time
    if time_elapsed < throttle_time:
        time.sleep(throttle_time - time_elapsed)
    response = requests.post(url, json=payload, headers=headers)
    if response.ok:
        score = response.json().get("score")
    else:
        print(f"Response not ok: {response.json()}")
    return score

def build_graph(graph: list) -> nx.Graph:
    G = nx.Graph()
    for triplet in graph:
        h, r, t = triplet
        G.add_edge(h, t, relation=r.strip())
    return G

def path_to_string(path: list) -> str: #Taken from utils.py
    result = ""
    for i, p in enumerate(path):
        if i == 0:
            h, r, t = p
            result += f"{h} -> {r} -> {t}"
        else:
            _, r, t = p
            result += f" -> {r} -> {t}"
    return result.strip()

def get_shortest_path(q_entity: list, t: str, graph: nx.Graph, include_all_paths: bool) -> list: #Taken from graph_utils.py
    h = q_entity[0]
    path = []
    try:
        path = nx.shortest_path(graph, h, t)
    except:
        path = [h, t] #Dummy path
    return [path]

def get_truth_paths(q_entity: list, a_entity: list, graph: nx.Graph, include_all_paths: bool = False) -> list: #Taken from graph utils
    '''
    Get shortest paths connecting question and answer entities.
    '''
    # Select paths
    paths = []
    for t in a_entity:
        paths += get_shortest_path(q_entity, t, graph, include_all_paths)
    # Add relation to paths
    result_paths = []
    for p_idx, p in enumerate(paths):
        tmp = []
        for i in range(len(p)-1):
            u = p[i]
            v = p[i+1]
            relation = "related" #Default dummy relation for dummy path
            if u in graph and v in graph[u] and 'relation' in graph[u][v]:
                relation = graph[u][v]['relation']
            tmp.append((u, relation, v))
        result_paths.append(tmp)
    return result_paths

def get_reasoning_paths(q_entity: list, a_entity: list, tuples: list):
    graph = build_graph(tuples)
    result_paths = get_truth_paths(q_entity, a_entity, graph)
    reasoning_paths = [path_to_string(path) for path in result_paths]
    return reasoning_paths

def get_top_k(query_ids, ent_pred, id2ent, triples, top_k):
    pred_ids = ent_pred.sort().indices[0, -top_k:].cpu().numpy()
    pred_entities = [id2ent[local_id] for local_id in pred_ids]
    query_entities = [id2ent[local_id] for local_id in query_ids]
    reasoning_paths = get_reasoning_paths(
        q_entity=query_entities, 
        a_entity=pred_entities,
        tuples=triples
    )
    docs = [
        {"title": ent, "content": path}
        for ent, path in zip(pred_entities, reasoning_paths)
    ]
    for path in reasoning_paths[::-1]:
        print(path)
    return docs

def convert_global_to_local(question_dict):
    g2l = {global_id: local_id for local_id, global_id in enumerate(question_dict["subgraph"]["entities"])}
    l2g = {v: k for k, v in g2l.items()}
    return g2l, l2g

def get_id_dicts():
    ent2id = {}
    with open(ENT2ID_FILE, "r") as f:
        ent2id = json.load(f)
    id2ent = {v: k for k, v in ent2id.items()}
    rel2id = {}
    with open(REL2ID_FILE, "r") as f:
        rel2id = json.load(f)
    id2rel = {v: k for k, v in rel2id.items()}
    return ent2id, id2ent, rel2id, id2rel


def update_subgraph(question_dict): #KG dataset already generates the ID files so no need to do it here
    # for stage in ["stage1", "stage2"]:
    #     stage_dir = os.path.join(PROCESSED_FOLDER, stage)
    #     if not os.path.isdir(stage_dir):
    #         os.makedirs(stage_dir)
    # global2local, local2global = convert_global_to_local(question_dict)
    id2ent = {}
    with open(ENTITIES_FILE, "r") as f:
        for global_id, line in enumerate(f.readlines()):
            id2ent[global_id] = line.strip()
    # Get ent2id
    # with open(ENTITIES_FILE, "r") as f:
    #     for global_id, line in enumerate(f.readlines()):
    #         if global_id in global2local.keys(): # Convert global ID to local_id
    #             ent2id[line.strip()] = global2local[global_id]
    #Save ent2id
    # with open(ENT2ID_FILE, "w") as f:
    #     f.write(f"{json.dumps(ent2id)}\n")
    # id2ent = {v: k for k, v in ent2id.items()}
    # Get rel2id
    id2rel = {}
    with open(RELATIONS_FILE, "r") as f:
        for global_id, line in enumerate(f.readlines()):
            id2rel[global_id] = line.strip()
    #Save rel2id
    # with open(REL2ID_FILE, "w") as f:
    #     f.write(f"{json.dumps(rel2id)}\n")
    # id2rel = {v: k for k, v in rel2id.items()}
    # Save kg.txt
    triples = [
        (id2ent[h], id2rel[r], id2ent[t])
        for h, r, t in question_dict["subgraph"]["tuples"]
    ]
    with open(KG_FILE, "w") as f:
        for trip in triples:
            f.write(f"{KG_DELIMITER.join(trip).strip()}\n")
    return triples

@hydra.main(
    config_path="config", config_name="stage3_qa_ircot_inference", version_base=None
)
def main(cfg: DictConfig, data_split="dev", top_k=5) -> None:
    qa_prompt_builder = QAPromptBuilder(cfg.qa_prompt)
    scores = []
    num_failures = 0
    with open(os.path.join(SRC_FOLDER, f"{data_split}.json"), "r") as f:
        for i, line in enumerate(tqdm(f.readlines())):
            question_dict = json.loads(line)
            triples = update_subgraph(question_dict) #Update to make sure we are focusing on relevant subgraph and query entities
            retriever = GFMRetriever.from_config(cfg) # Currently have to reinit each time for updated graph files
            ent2id, id2ent, rel2id, id2rel = get_id_dicts()
            g2l, l2g = convert_global_to_local()
            query_ids = [g2l[question_dict["entities"][0]]] # Just get the local ID of the first query entity
            ent_pred = retriever.retrieve(question_dict["question"], query_ids)
            docs = get_top_k(query_ids, ent_pred, id2ent, triples, top_k)
            messages = qa_prompt_builder.build_input_prompt(question_dict["question"], docs)
            score = evaluate_llm(messages, question_dict["answer"])
            if score > 0:
                scores.append(score)
            else:
                num_failures += 1
            print(f"Mean of scores so far: {np.mean(scores)}")
    print(f"Final mean of scores: {np.mean(scores)}")
    print(f"Final num failures: {num_failures}")

main()
