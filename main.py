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
DST_FOLDER = os.path.join("data", "hotpotqa_test")
PROCESSED_FOLDER = os.path.join(DST_FOLDER, "processed")
RAW_FOLDER = os.path.join(DST_FOLDER, "raw")
SRC_FOLDER = os.path.join("..", "GNN-RAG", "gnn", "data", "finqa-debug")

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
        import pdb; pdb.set_trace()
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

def process_id_dict(entities_file, ent2id_file):
    ent2id = {}
    with open(entities_file, "r") as f:
        for i, line in enumerate(f.readlines()):
            ent2id[line.strip()] = i
    with open(ent2id_file, "w") as f:
        f.write(f"{json.dumps(ent2id)}\n")
    id2ent = {v: k for k, v in ent2id.items()}
    return ent2id, id2ent

def update_subgraph(question_dict,
               entities_file=os.path.join(SRC_FOLDER, "entities.txt"),
               relations_file=os.path.join(SRC_FOLDER, "relations.txt"),
               ent2id_file=os.path.join(PROCESSED_FOLDER, "stage2", "ent2id.json"),
               rel2id_file=os.path.join(PROCESSED_FOLDER, "stage2", "rel2id.json"),
               kg_file=os.path.join(PROCESSED_FOLDER, "stage1", "kg.txt"),
               doc2ent_file=os.path.join(PROCESSED_FOLDER, "stage1", "document2entities.json"),
               corpus_file=os.path.join(RAW_FOLDER, "dataset_corpus.json")):
    subgraph = question_dict["subgraph"]
    for start_folder in [PROCESSED_FOLDER, RAW_FOLDER]:
        for stage in ["stage1", "stage2"]:
            stage_dir = os.path.join(start_folder, stage)
            if not os.path.isdir(stage_dir):
                os.makedirs(stage_dir)
    ent2id, id2ent = process_id_dict(entities_file, ent2id_file)
    rel2id, id2rel = process_id_dict(relations_file, rel2id_file)
    all_entities = [id2ent[my_id] for my_id in subgraph["entities"]]
    query_entities = [id2ent[my_id] for my_id in question_dict["entities"]]
    triples = [
        [id2ent[h], id2rel[r], id2ent[t]]
        for h, r, t in subgraph["tuples"]
    ]
    with open(kg_file, "w") as f: # Get kg.txt tuples from subgraph tuples
        for trip in triples:
            trip = KG_DELIMITER.join(trip).strip()
            f.write(f"{trip}\n")
    # We could get away with commenting out this stuff if we don't need it in our case
    with open(doc2ent_file, "w") as f: # Save document2entities file. Since we dont have docs, let the doc title just be the entity
        document2entities = { ent: [ent] for ent in all_entities }
        f.write(json.dumps(document2entities))
    with open(corpus_file, "w") as f: # Each document for an entity is just the shortest path associated with it
        dataset_corpus = {}
        reasoning_paths = get_reasoning_paths(
           q_entity=query_entities, 
           a_entity=all_entities,
           tuples=triples
        )
        for i, ent in enumerate(all_entities):
            dataset_corpus[ent] = reasoning_paths[i]
        f.write(json.dumps(dataset_corpus))

def get_local_query_entities(question_dict):
    all_entities = question_dict["subgraph"]["entities"]
    query_entities = question_dict["entities"]
    local_ids = []
    for i, ent_id in enumerate(all_entities):
        if ent_id in query_entities:
            local_ids.append(i)
    return local_ids
        
@hydra.main(
    config_path="config", config_name="stage3_qa_ircot_inference", version_base=None
)
def main(cfg: DictConfig, data_split="dev", top_k=5) -> None:
    qa_prompt_builder = QAPromptBuilder(cfg.qa_prompt)
    scores = []
    with open(os.path.join(SRC_FOLDER, f"{data_split}.json"), "r") as f:
        for line in tqdm(f.readlines()):
            start_time = time.time()
            question_dict = json.loads(line)
            update_subgraph(question_dict) #Update to make sure we are focusing on relevant subgraph and query entities
            subgraph_time = time.time()
            print(f"Update subgraph time: {subgraph_time - start_time}")
            retriever = GFMRetriever.from_config(cfg) # Currently have to reinit each time for updated graph files
            init_time = time.time()
            print(f"Init time: {init_time - subgraph_time}")
            query_entities = get_local_query_entities(question_dict)
            docs = retriever.retrieve(question_dict["question"], query_entities, top_k=5)
            retrieval_time = time.time()
            print(f"Retrieval time: {retrieval_time - init_time}")
            messages = qa_prompt_builder.build_input_prompt(question_dict["question"], docs)
            score = evaluate_llm(messages, question_dict["answer"])
            eval_time = time.time()
            print(f"Eval time: {eval_time - retrieval_time}")
            scores.append(score)
            print(f"Score: {score}")
    print(f"Mean of scores: {np.mean(scores)}")

main()
