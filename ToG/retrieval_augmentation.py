from tqdm import tqdm
import argparse
import random
from wiki_func import *
from client import *
import os

QUESTION_TEMPLATE = """
        Here comes your question:
        Question: {q}
        Document: {d}
        Answer:
    """


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="webqsp", help="choose the dataset."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="the max length of LLMs output.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="the temperature in exploration stage.",
    )
    parser.add_argument(
        "--LLM_type", type=str, default="gpt-3.5-turbo-1106", help="base LLM model."
    )
    parser.add_argument(
        "--openai_api_keys",
        type=str,
        default="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",
        help="if the LLM_type is gpt-3.5-turbo or gpt-4, you need add your own openai api keys.",
    )
    parser.add_argument(
        "--addr_list",
        type=str,
        default="./server_urls.txt",
        help="The address of the Wikidata service.",
    )
    parser.add_argument(
        "--few_shot_path",
        type=str,
        default="./rag_prompt.txt",
        help="Path to few shot data.",
    )
    parser.add_argument(
        "--result_save_path", type=str, default="./rag_results"
    )
    args = parser.parse_args()
    return args


def load_server_addresses(file_path):
    with open(file_path, "r") as f:
        return [addr.strip() for addr in f.readlines()]


def process_data(
    data,
    args,
    wiki_client: MultiServerWikidataQueryClient,
    question_string: str,
    few_shot_prefix: str,
):
    question = data[question_string]
    topic_entity = list(data["qid_topic_entity"])
    cluster_chain_of_entities = []
    pre_relations = ([],)
    pre_heads = [-1] * len(topic_entity)
    flag_printed = False

    related_passage = wiki_client.query_all(
        "get_wikipedia_page", topic_entity[0]
    )
    related_passage = "\n".join(related_passage)
    
    not_found = True if 'Not Found!' in related_passage else False    
    
    llm_input = few_shot_prefix + QUESTION_TEMPLATE.format(
        q=question, d=related_passage
    )
    response = run_llm(
        prompt=llm_input,
        temperature=args.temperature,
        max_tokens=args.max_length,
        engine=args.LLM_type,
        opeani_api_keys=args.openai_api_keys,
    )
    return {
        "results": response,
        "question": question,
        "llm_input": llm_input,
        "not_found": not_found,
        "topic_entity": topic_entity,
        
    }


def main(args):
    datas, question_string = prepare_dataset(args.dataset)
    server_addresses = load_server_addresses(args.addr_list)
    print(f"Server addresses: {server_addresses}")
    wiki_client = MultiServerWikidataQueryClient(server_addresses)

    with open(args.few_shot_path, "r") as f:
        few_shot_data = "\n".join(f.readlines())

    # # Clear the result file
    # with open(args.result_save_path, "w") as f:
    #     pass
    os.makedirs(args.result_save_path, exist_ok=True)

    results = []
    for i, data in tqdm(enumerate(datas)):
        result = process_data(
            data,
            args,
            wiki_client,
            question_string=question_string,
            few_shot_prefix=few_shot_data,
        )
        # results.append(result)
        with open(os.path.join(args.result_save_path, f"{i}.json"), "w") as f:
            f.write(json.dumps(result) + '\n')


if __name__ == "__main__":
    args = parse_args()
    main(args)
