import logging
import os

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
import gfmrag
from gfmrag.llms import BaseLanguageModel
from gfmrag.prompt_builder import QAPromptBuilder

from gfmrag import GFMRetriever

logger = logging.getLogger(__name__)


@hydra.main(
    config_path="config", config_name="stage3_qa_ircot_inference", version_base=None
)
def main(cfg: DictConfig) -> None:
    output_dir = HydraConfig.get().runtime.output_dir
    logger.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Output directory: {output_dir}")

    current_query = "Who is the president of France?"
    retriever = GFMRetriever.from_config(cfg)
    docs = retriever.retrieve(current_query, top_k=1)
    import pdb; pdb.set_trace()

    llm = instantiate(cfg.llm)
    qa_prompt_builder = QAPromptBuilder(cfg.qa_prompt)

    message = qa_prompt_builder.build_input_prompt(current_query, docs)
    answer = llm.generate_sentence(message)  # Answer: "Emmanuel Macron"
    print(answer)

main()
