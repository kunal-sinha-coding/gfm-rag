import logging
import os
import time
import json
import requests
import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
import re

import yaml

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

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

model_name = "rmanluo/RoG"
device = "cuda" if torch.cuda.is_available() else "cpu"
DATA_NAME = "webqsp"
PROCESSED_FOLDER = os.path.join("data", DATA_NAME, "processed")
if not os.path.isdir(PROCESSED_FOLDER):
    for stage in ["stage1", "stage2"]:
        os.makedirs(os.path.join(PROCESSED_FOLDER, stage), exist_ok=True)
SRC_FOLDER = os.path.join("..", "GNN-RAG", "gnn", "data", DATA_NAME)
ENT_NAMES_FILE = os.path.join("..", "GNN-RAG", "gnn", "entities_names.json")
ENTITIES_FILE = os.path.join(SRC_FOLDER, "entities.txt")
RELATIONS_FILE = os.path.join(SRC_FOLDER, "relations.txt")
ENT2ID_FILE = os.path.join(PROCESSED_FOLDER, "stage2", "ent2id.json")
REL2ID_FILE = os.path.join(PROCESSED_FOLDER, "stage2", "rel2id.json")
KG_FILE = os.path.join(PROCESSED_FOLDER, "stage1", "kg.txt")

CACHE_DIR = "../cache_dir"
MAX_NEW_TOKENS = 2048
LLAMA_TOKEN = os.getenv("LLAMA_TOKEN")
LLAMA_PROMPT = (
    '''
    [INST] <<SYS>>
    <</SYS>>
    {prompt}
    {context}
    [/INST]
    '''
)

LMUNIT_TEST = "Is the response correct? Groundtruth: {groundtruth}"

def generate(prompt, llm_model, tokenizer, context=""): # Comes from GNN-RAG generate_dataset_from_hf.py file
    #llm_prompt = [tokenizer.apply_chat_template(
    #    messages,
    #    tokenize=False,
    #    add_generation_prompt=True
    #)]
    llm_prompt = LLAMA_PROMPT.format(prompt=prompt, context=context)
    print(llm_prompt)
    inputs = tokenizer(llm_prompt, return_tensors="pt").to(device)
    inputs_len = inputs.input_ids.size(-1)
    outputs = llm_model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    response = tokenizer.decode(outputs[0][inputs_len:], skip_special_tokens=True)
    return response

def get_groundtruth(question_dict):
    answer_key = "answer" if "answer" in question_dict else "answers"                                                  
    groundtruth = []
    try:
        groundtruth = question_dict[answer_key]
        if len(groundtruth) > 0 and isinstance(groundtruth[0], dict):
            answers = []
            for current in groundtruth:
                answers.append(current["text"] if current["text"] else current["kb_id"])
            groundtruth = answers
        groundtruth = [current.strip() for current in groundtruth]
    except:                                                                                                            
        print("Failed on: ", question_dict[answer_key])                                                                
    return groundtruth

def format_prediction(prediction):
    if "1." in prediction:
        prediction = prediction[prediction.index("1."):]
    pred_formatted = re.sub(r'\d+\.', '', prediction).strip().lower().split("\n")
    return [pred.strip() for pred in pred_formatted if pred not in ["", "?"]]
    
def evaluate_llm(prompt, question_dict, long_answer, llm_model, tokenizer,
        throttle_time=1, table_name=None, include_reasoning_paths=True):
    # To run long context, set max_num_paths=2000 and include_reasoning_paths=True
    # To run llm-only, set inlcude_reasoning_paths=False
    correct, score = 0, 0
    groundtruth = get_groundtruth(question_dict)
    start_time = time.time()
    try:
        prediction = generate(prompt, llm_model, tokenizer).strip()
    except:
        print("Failed on generate sentence")
        print(f"Curr input: {prompt}")
        return 0, 0
    if long_answer:
        unit_test = f"Is the response correct? Groundtruth: {groundtruth[i]}"
        url = "https://api.contextual.ai/v1/lmunit"
        lm_unit_api_key = os.getenv("LM_UNIT_API_KEY")
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {lm_unit_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "query": question_dict["question"][i],
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
        # Treat "scores" as h1
        pred_formatted = format_prediction(prediction)
        print(f"PREDICTION: {prediction}")
        print(f"PRED_FORMATTED: {pred_formatted}")
        for j, pred in enumerate(pred_formatted):
            for gt in groundtruth:
                gt_formatted = gt.strip().lower()
                if gt_formatted == pred:#pred in gt_formatted or gt_formatted in pred:
                    correct = 1
                    if j == 0:
                        score = 1
    return correct, score

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

def get_shortest_path(q_entity: list, t: str, graph: nx.Graph, max_new_paths: int) -> list: #Taken from graph_utils.py
    paths = []
    for h in q_entity:
        try:
            for p in nx.all_shortest_paths(graph, h, t):
                paths.append(p)
            if len(paths) > max_new_paths:
                return paths
        except:
            continue
    #If no path found, return a dummy path
    if len(paths) == 0:
        h = q_entity[0] if len(q_entity) > 0 else t
        paths = [[h, t]]
    return paths

def get_truth_paths(q_entity: list, a_entity: list, graph: nx.Graph, max_new_paths: int) -> list: #Taken from graph utils
    '''
    Get shortest paths connecting question and answer entities.
    '''
    # Select paths
    paths = []
    for t in a_entity:
        paths += get_shortest_path(q_entity, t, graph, max_new_paths)
    # Add relation to paths
    result_paths = []
    for p_idx, p in enumerate(paths):
        tmp = []
        for i in range(len(p)-1):
            u = p[i]
            v = p[i+1]
            # Get rid of dummy relation
            if u in graph and v in graph[u] and "relation" in graph[u][v]:
                #for rel in graph[u][v].values(): #ToDo: handle the different possible paths from multigraph. But idk if this fixes root issue
                #rel = graph[u][v][0]
                relation = graph[u][v]["relation"]
                tmp.append((u, relation, v))
        result_paths.append(tmp)
    return result_paths

def get_reasoning_paths(q_entity: list, a_entity: list, tuples: list):
    graph = build_graph(tuples)
    result_paths = get_truth_paths(q_entity, a_entity, graph, max_new_paths=2000)
    reasoning_paths = [path_to_string(path) for path in result_paths]
    return reasoning_paths

def get_top_k(query_ids, ent_pred, id2ent, triples, top_k):
    sorted_indices = ent_pred[0].sort().indices
    pred_ids = sorted_indices[-top_k:].cpu().numpy()
    pred_entities = [id2ent[local_id] for local_id in pred_ids]
    query_entities = [id2ent[local_id] for local_id in query_ids]
    reasoning_paths = get_reasoning_paths(
        q_entity=query_entities, 
        a_entity=pred_entities,
        tuples=triples
    )
    #docs = [
    #    {"title": "path", "content": path}
    #    for path in reasoning_paths
    #]
    #for path in reasoning_paths[::-1]:
        #print(path)
    #return docs
    return reasoning_paths

#def convert_global_to_local(entities):
#    g2l = {global_id: local_id for local_id, global_id in enumerate(entities)}
#    l2g = {v: k for k, v in g2l.items()}
#    return g2l, l2g

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
    with open(ENT_NAMES_FILE, 'r') as f:
        entities_names = json.load(f)
    global_id2ent = {}
    with open(ENTITIES_FILE, "r") as f:
        for global_id, line in enumerate(f.readlines()):
            ent = line.strip()
            ent = entities_names[ent] if ent in entities_names else ent
            global_id2ent[global_id] = ent
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
    global_id2rel = {}
    with open(RELATIONS_FILE, "r") as f:
        for global_id, line in enumerate(f.readlines()):
            global_id2rel[global_id] = line.strip()
    #Save rel2id
    # with open(REL2ID_FILE, "w") as f:
    #     f.write(f"{json.dumps(rel2id)}\n")
    # id2rel = {v: k for k, v in rel2id.items()}
    # Save kg.txt
    query_entities = [global_id2ent[global_id] for global_id in question_dict["entities"]]
    triples = [
        (global_id2ent[h], global_id2rel[r], global_id2ent[t])
        for h, r, t in question_dict["subgraph"]["tuples"]
    ]
    with open(KG_FILE, "w") as f:
        for trip in triples:
            f.write(f"{KG_DELIMITER.join(trip).strip()}\n")
    return triples, query_entities

def get_gnnrag_prompt(question, reasoning_paths, long_answer):
    paths = "\n".join(reasoning_paths)
    if long_answer:
        return f"Based on the reasoning paths, please answer the given question in one sentence.\nReasoning paths: {paths}\nQuestion: {question}\nAnswer:"
    return f"Based on the reasoning paths, please answer the given question. Please keep the answer as simple as possible and return all the possible answers as a list.\n Reasoning paths: {paths}\nQuestion: {question}\nAnswer:"


@hydra.main(
    config_path="config", config_name="stage3_qa_ircot_inference", version_base=None
)
def main(cfg: DictConfig, data_split="test", top_k=10, long_answer=False) -> None:
    qa_prompt_builder = QAPromptBuilder(cfg.qa_prompt)
    cfg.dataset.data_name = DATA_NAME # Overwrite yaml file with correct data name
    num_failures = 0
    scores = []
    recall = []
    llm_model = LlamaForCausalLM.from_pretrained(
        model_name,
        cache_dir=CACHE_DIR,
	token=LLAMA_TOKEN
    ).to(device, dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=False,
        token=LLAMA_TOKEN
    )
    with open(os.path.join(SRC_FOLDER, f"{data_split}.json"), "r") as f:
        for i, line in enumerate(tqdm(f.readlines())):
            question_dict = json.loads(line)
            triples, query_entities = update_subgraph(question_dict) #Update to make sure we are focusing on relevant subgraph and query entities
            retriever = GFMRetriever.from_config(cfg) # Currently have to reinit each time for updated graph files
            ent2id, id2ent, rel2id, id2rel = get_id_dicts()
            #g2l, l2g = convert_global_to_local(question_dict["subgraph"]["entities"])
            query_ids = [ent2id[ent] for ent in query_entities]
            ent_pred = retriever.retrieve(question_dict["question"], query_ids)
            #print(f"Question: ", question_dict["question"])
            #print(f"Answer: ", question_dict["answer"])
            #print(f"Paths: ", question_dict["paths"])
            #try:
                #sorted_indices = ent_pred[0].sort().indices
                #answer_node = get_answer_node(question_dict)
                #correct_idx = sorted_indices.size(0) - 1 - (sorted_indices == ent2id[answer_node]).float().argmax()
                #recall.append(correct_idx.item() < top_k)
                #print(correct_idx.item())
                #print(f"Retrieval recall: {np.mean(recall)}")
            #except:
                #num_failures += 1
            reasoning_paths = get_top_k(query_ids, ent_pred, id2ent, triples, top_k)
            prompt = get_gnnrag_prompt(question_dict["question"], reasoning_paths, long_answer)
            #messages = qa_prompt_builder.build_input_prompt(question_dict["question"], docs)
            correct, score = evaluate_llm(prompt, question_dict, long_answer, llm_model, tokenizer)
            if score > 0:
                scores.append(score)
            #else:
                #num_failures += 1
            print(f"Mean of scores so far: {np.mean(scores)}")
    print(f"Final mean of scores: {np.mean(scores)}")
    print(f"Final num failures: {num_failures}")
    print(f"Final retrieval recall: {np.mean(recall)}")

if __name__ == '__main__':
    main()
