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
SRC_FOLDER = os.path.join("..", "GNN-RAG", "gnn", "data", "CWQ")
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

def get_groundtruth(self, question_dict):
    groundtruth = []
    for i in range(len(question_dict["question"])):
        gt = [""]
        answer_key = "answer" if "answer" in question_dict else "answers"                                                  
        try:                                                                                                               
            gt = question_dict[answer_key][i]
            #if len(gt) > 0 and isinstance(gt[0], dict):
            #    answers = []
            #    for current in gt:
            #        answers.append(current["text"] if current["text"] else current["kb_id"])
            #    gt = answers
            gt = [current.strip() for current in gt]
        except:                                                                                                            
            print("Failed on: ", question_dict[answer_key])                                                                
        groundtruth.append(gt)
    return groundtruth

def format_prediction(self, prediction):
    if "1." in prediction:
        prediction = prediction[prediction.index("1."):]
    pred_formatted = re.sub(r'\d+\.', '', prediction).strip().lower().split("\n")
    return [pred.strip() for pred in pred_formatted if pred not in ["", "?"]]
    
def evaluate_llm(self, question_dict, long_answer=False, 
        throttle_time=1, table_name=None, max_num_paths=2000, include_reasoning_paths=True):
    # To run long context, set max_num_paths=2000 and include_reasoning_paths=True
    # To run llm-only, set inlcude_reasoning_paths=False
    all_input, _ = self.input_builder.process_input_batch(
        question_dict, max_num_paths=max_num_paths,
        include_reasoning_paths=include_reasoning_paths
    )
    correct = [0 for inp in all_input]
    scores = [0 for inp in all_input]
    groundtruth = self.get_groundtruth(question_dict)
    for i, curr_input in enumerate(all_input):
        start_time = time.time()
        try:
            prediction = self.llm_model.generate_sentence(curr_input).strip()
        except:
            print("Failed on generate sentence")
            print(f"Curr input: {curr_input}")
            continue
        answer_key = "answer" if "answer" in question_dict else "answers"
        groundtruth = self.get_groundtruth(question_dict)
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
                if score:
                    scores[i] = score
                    correct[i] = score
                else:
                    print(f"Response not ok: {response.json()}")
            #table_data = self.llm_output_table.data
            #iteration = table_data[-1][0] + 1 if len(table_data) > 0 else 0
            #self.llm_output_table.add_data(
            #    iteration, curr_input, prediction, unit_test, correct[i]
            #)
            #self.llm_output_table = wandb.Table(
            #    columns=self.llm_output_table_cols, data=self.llm_output_table.data
            #)
            #if table_name:
                #wandb.log({table_name: self.llm_output_table})
        else:
            # Treat "scores" as h1
            pred_formatted = self.format_prediction(prediction)
            for j, pred in enumerate(pred_formatted):
                for gt in groundtruth[i]:
                    gt_formatted = gt.strip().lower()
                    if gt_formatted == pred:#pred in gt_formatted or gt_formatted in pred:
                        correct[i] = 1
                        if j == 0:
                            scores[i] = 1
    return correct, scores

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
    paths = []
    for h in q_entity:
        if include_all_paths:
            try:
                for p in nx.all_shortest_paths(graph, h, t):
                    paths.append(p)
            except:
                continue
        else:
            try:
                p = nx.shortest_path(graph, h, t)
                paths.append(p)
            except:
                continue
    #If no path found, return a dummy path
    if len(paths) == 0:
        h = q_entity[0] if len(q_entity) > 0 else t
        paths = [[h, t]]
    return paths

def get_truth_paths(q_entity: list, a_entity: list, graph: nx.Graph, include_all_paths: bool) -> list: #Taken from graph utils
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
    result_paths = get_truth_paths(q_entity, a_entity, graph, include_all_paths=False)
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
    global_id2ent = {}
    with open(ENTITIES_FILE, "r") as f:
        for global_id, line in enumerate(f.readlines()):
            global_id2ent[global_id] = line.strip()
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

def get_gnnrag_prompt(question, reasoning_paths):
    messages = [
        {"role": "user", "content": f"Based on the reasoning paths, please answer the given question in one sentence.\nReasoning paths: {reasoning_paths}\nQuestion: {question}\nAnswer:"}
    ]
    return messages

@hydra.main(
    config_path="config", config_name="stage3_qa_ircot_inference", version_base=None
)
def main(cfg: DictConfig, data_split="dev", top_k=10) -> None:
    qa_prompt_builder = QAPromptBuilder(cfg.qa_prompt)
    num_failures = 0
    scores = []
    recall = []
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
            messages = get_gnnrag_prompt(question_dict["question"], reasoning_paths)
            #messages = qa_prompt_builder.build_input_prompt(question_dict["question"], docs)
            score = evaluate_llm(messages, question_dict["answer"] if "answer" in question_dict else answer_node)
            print(score)
            import pdb; pdb.set_trace()
            if score > 0:
                scores.append(score)
            #else:
                #num_failures += 1
            print(f"Mean of scores so far: {np.mean(scores)}")
    print(f"Final mean of scores: {np.mean(scores)}")
    print(f"Final num failures: {num_failures}")
    print(f"Final retrieval recall: {np.mean(recall)}")

main()
