"""
batch_selfinstruct_generate.py

run:
python -m generate_instruction generate_instruction_following_data \
  --output_dir ./ \
  --num_instructions_to_generate 10 \
  --model_name="text-davinci-003" \
"""
import time
import json
import os
import random
import re
import string
from functools import partial
from multiprocessing import Pool

import numpy as np
import tqdm
from rouge_score import rouge_scorer
import utils

import fire


def encode_prompt(prompt_instructions):
    """
        I love you.
        나는 너를 사랑해.
        
        Encode multiple prompt instructions into a single string.
        


    """
    prompt = open("./prompt.txt").read() + "\n"

    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<noinput>" if input.lower() == "" else input
        prompt += f"###\n"
        prompt += f"{idx + 1}. Instruction: {instruction}\n"
        prompt += f"{idx + 1}. Input:\n{input}\n"
        prompt += f"{idx + 1}. Output:\n{output}\n"
    prompt += f"###\n"
    prompt += f"{idx + 2}. Instruction:"
    return prompt


def post_process_gpt3_response(num_prompt_instructions, response):
    """
        Post-process the response from GPT-3.
        ko: GPT-3의 응답을 후처리한다.
    """
    if response is None:
        return []
    raw_instructions = f"{num_prompt_instructions+1}. Instruction:" + response["text"]
    raw_instructions = re.split("###", raw_instructions)
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        # ko: 마지막 예제가 잘릴 수 있으므로 버린다.
        if idx == len(raw_instructions) - 1 and response["finish_reason"] == "length":
            continue

        # 
        idx += num_prompt_instructions + 1

        # split the instruction into instruction, input, output
        # ko: instruction, input, output으로 나눈다.
        splitted_data = re.split(f"{idx}\.\s+(Instruction|Input|Output):", inst)
        if len(splitted_data) != 7:
            continue
        else:
            inst = splitted_data[2].strip()
            input = splitted_data[4].strip()
            input = "" if input.lower() == "<noinput>" else input
            output = splitted_data[6].strip()


        # filter out too short or too long instructions
        # ko: 너무 짧거나 너무 긴 instruction은 제외한다.
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue

        # filter based on keywords that are not suitable for language models.
        # ko: 언어 모델에 적합하지 않은 키워드를 기반으로 필터링한다.
        blacklist = [
            "image",
            "images",
            "graph",
            "graphs",
            "picture",
            "pictures",
            "file",
            "files",
            "map",
            "maps",
            "draw",
            "plot",
            "go to",
            "video",
            "audio",
            "music",
            "flowchart",
            "diagram",
        ]
        blacklist += []
        if any(find_word_in_string(word, inst) for word in blacklist):
            continue
        
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # ko: 모델은 일부 기존 지시문에 "프로그램 작성"을 추가하는 경향이 있으므로 많은 지시문이 생성된다.
        # And it's a bit comfusing whether the model need to write a program or directly output the result.
        # ko: 모델이 프로그램을 작성해야하는지 결과를 직접 출력해야하는지 약간 혼란스럽다.
        # Here we filter them out.
        # ko: 여기서 이것들을 필터링한다.
        # Note this is not a comprehensive filtering for all programming instructions.
        # ko: 이것은 모든 프로그래밍 지시문에 대한 종합적인 필터링이 아니다.
        if inst.startswith("Write a program"):
            continue
        # filter those starting with punctuation
        # ko: 구두점으로 시작하는 것을 필터링한다.
        if inst[0] in string.punctuation:
            continue
        # filter those starting with non-english character
        # ko: 영어가 아닌 문자로 시작하는 것을 필터링한다.
        if not inst[0].isascii():
            continue
        instructions.append({"instruction": inst, "input": input, "output": output})
    return instructions


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def generate_instruction_following_data(
    output_dir="./",
    seed_tasks_path="./seed_tasks.jsonl",
    num_instructions_to_generate=100,
    model_name="text-davinci-003",
    num_prompt_instructions=3,
    request_batch_size=5,
    temperature=1.0,
    top_p=1.0,
    num_cpus=16,
):
    # load the human-written seed instructions
    # ko: seed_tasks.jsonl 파일을 읽어서 seed_tasks에 저장

    seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
    seed_instruction_data = [
        {"instruction": t["instruction"], "input": t["instances"][0]["input"], "output": t["instances"][0]["output"]}
        for t in seed_tasks
    ]
    print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")

    # create output directory
    # ko: output_dir이 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    request_idx = 0
    
    
    # load the LM-generated instructions
    # ko: regen.json 파일을 읽어서 machine_instruction_data에 저장
    # we will use these to compute the similarity between the generated instructions and the seed instructions
    machine_instruction_data = []
    if os.path.exists(os.path.join(output_dir, "regen.json")):
        machine_instruction_data = utils.jload(os.path.join(output_dir, "regen.json"))
        print(f"Loaded {len(machine_instruction_data)} machine-generated instructions")

    # we will use the ROUGE-L score to measure the similarity between the generated instructions and the seed instructions
    # ko: ROUGE-L score를 사용하여 생성된 지침과 seed 지침 사이의 유사성을 측정
    # similarities = {}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # now let's generate new instructions!
    # ko: 이제 새로운 지침을 생성하자!
    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
    if machine_instruction_data:
        progress_bar.update(len(machine_instruction_data))

    # first we tokenize all the seed instructions and generated machine instructions
    # ko: 먼저 모든 seed 지침과 생성된 기계 지침을 토큰화
    all_instructions = [d["instruction"] for d in seed_instruction_data] + [
        d["instruction"] for d in machine_instruction_data
    ]
    # ko: all_instruction_tokens에는 seed_instruction_data와 machine_instruction_data의 instruction을 tokenize한 결과가 저장
    all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in all_instructions]

    # we will use the ROUGE-L score to measure the similarity between the generated instructions and the seed instructions
    # ko: ROUGE-L score를 사용하여 생성된 지침과 seed 지침 사이의 유사성을 측정
    while len(machine_instruction_data) < num_instructions_to_generate:
        request_idx += 1

        # generate a batch of requests
        # ko: 요청의 일괄 처리를 생성
        batch_inputs = []
        for _ in range(request_batch_size):
            # only sampling from the seed tasks
            prompt_instructions = random.sample(seed_instruction_data, num_prompt_instructions)
            prompt = encode_prompt(prompt_instructions)
            batch_inputs.append(prompt)

        # send the requests to the API
        # ko: API에 요청을 보냄

        # hard-code to maximize the length. the requests will be automatically adjusted
        # ko: 최대 길이로 하드 코딩. 요청은 자동으로 조정됨
        decoding_args = utils.OpenAIDecodingArguments(
            temperature=temperature,
            n=1,
            max_tokens=3072,  # hard-code to maximize the length. the requests will be automatically adjusted
            top_p=top_p,
            stop=["\n20", "20.", "20."],
        )
        
        
        request_start = time.time()

        # ko: openai_completion 함수를 통해 GPT-3 API를 호출
        results = utils.openai_completion(
            prompts=batch_inputs,
            model_name=model_name,
            batch_size=request_batch_size,
            decoding_args=decoding_args,
            logit_bias={"50256": -100},  # prevent the <|endoftext|> token from being generated
        )
        request_duration = time.time() - request_start

        process_start = time.time()

        # post-process the results
        # ko: 결과를 후처리
        instruction_data = []
        for result in results:
            new_instructions = post_process_gpt3_response(num_prompt_instructions, result)
            instruction_data += new_instructions

        total = len(instruction_data)
        keep = 0
        for instruction_data_entry in instruction_data:
            # computing similarity with the pre-tokenzied instructions
            # ko: 토큰화된 지침과 유사성 계산
            new_instruction_tokens = scorer._tokenizer.tokenize(instruction_data_entry["instruction"])
            with Pool(num_cpus) as p:
                rouge_scores = p.map(
                    partial(rouge_scorer._score_lcs, new_instruction_tokens),
                    all_instruction_tokens,
                )
            rouge_scores = [score.fmeasure for score in rouge_scores]
            most_similar_instructions = {
                all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
            }
            if max(rouge_scores) > 0.7:
                continue
            else:
                keep += 1
            instruction_data_entry["most_similar_instructions"] = most_similar_instructions
            instruction_data_entry["avg_similarity_score"] = float(np.mean(rouge_scores))
            machine_instruction_data.append(instruction_data_entry)
            all_instructions.append(instruction_data_entry["instruction"])
            all_instruction_tokens.append(new_instruction_tokens)
            progress_bar.update(1)
        process_duration = time.time() - process_start
        print(f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s")
        print(f"Generated {total} instructions, kept {keep} instructions")
        utils.jdump(machine_instruction_data, os.path.join(output_dir, "regen.json"))


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
