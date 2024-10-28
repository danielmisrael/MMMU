import os
import sys
import json
import torch
import yaml
import re
import ast
from PIL import Image
from tqdm import tqdm
# from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

from transformers import ChameleonForConditionalGeneration
from transformers import (
    ChameleonImageProcessor,
    ChameleonProcessor,
)
from transformers import LlamaTokenizerFast
from datasets import load_dataset

# Configuration
if len(sys.argv) == 3:
    MODEL = sys.argv[1]
    MODE = sys.argv[2]
    SETTING = sys.argv[3]
else:
    print("Usage: python script.py [MODEL] [MODE], default: python script.py llava-onevision-qwen2-7b-si-hf direct vision")
    MODEL = 'chlm_7b_new_hf'
    MODE = 'direct'
    # SETTING = 'vision'
    SETTING = 'standard'

MAX_RETRY = 5
NUM = 1730
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Adjust this if needed

# Load processor and model
path = "/localhome/data/ckpts/" + MODEL
tokenizer = LlamaTokenizerFast(
            tokenizer_file=os.path.join('/localhome/data/ckpts/chameleon/', "tokenizer/text_tokenizer_modified.json"), legacy=False
        )

tokenizer.sep_token_id = 8710  # assign <reserved08706> to sep so that we can append it after input text
tokenizer.pad_token_id = 1  # assing <pad> to special pad_token
image_processor = ChameleonImageProcessor()
processor = ChameleonProcessor(image_processor=image_processor, tokenizer=tokenizer)
model = ChameleonForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="auto") # #TODO Why is this not working?

# Load prompt configuration
with open("prompts.yaml", "r") as file:
    prompt_config = yaml.safe_load(file)[MODE]

def replace_images_tokens(input_string):
    for i in range(1, 8):
        question_text = f"<image {i}>"
        query_text = "<image>"
        if question_text in input_string:
            input_string = input_string.replace(question_text, query_text)
    return input_string

def parse_options(options):
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
    return choices_str

def construct_prompt(doc):
    question = doc["question"]
    parsed_options = parse_options(ast.literal_eval(str(doc["options"])))
    question = f"{question}\n{parsed_options}\n{prompt_config['standard']}"
    return question

def mmmu_doc_to_text(doc):
    question = construct_prompt(doc)
    return replace_images_tokens(question)

def origin_mmmu_doc_to_visual(doc):
    visual = []
    for i in range(1,8):
        if not doc[f'image_{i}']:
            break
        visual.append(doc[f'image_{i}'])
    return visual

def vision_mmmu_doc_to_visual(doc):
    return [doc['image']]

def process_prompt(data):
    if SETTING == 'standard':
        prompt = mmmu_doc_to_text(data)
        images = origin_mmmu_doc_to_visual(data)
    elif SETTING == 'vision':
        prompt = prompt_config['vision']
        images = vision_mmmu_doc_to_visual(data)
        
    return (prompt, images)

def save_results_to_file(results, output_path):
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for output, data in results:
            data['response'] = output
            data = {k: v for k, v in data.items() if not k.startswith('image')}
            json.dump(data, outfile, ensure_ascii=False)
            outfile.write('\n')

def run_and_save():
    
    if SETTING == 'standard':
        setting = 'standard (4 options)'
    elif SETTING == 'vision':
        setting = 'vision'
    dataset = load_dataset('MMMU/MMMU_Pro', setting, split='test').select(range(NUM))

    def process_and_save_part(part_data, part_name):
        print(f"Begin processing {part_name}")
        output_path = f"./output/{MODEL}_{part_name}_{MODE}.jsonl"
        results = []
        if os.path.exists(output_path):
            print(f"Loaded existing results for {part_name}")
        else:
            for idx, data in enumerate(tqdm(part_data, desc=f"Processing {part_name}"), start=1):
                prompt, images = process_prompt(data)
                if SETTING == 'vision':
                    prompt += "<image>"
                    
                # Hack to fix processing bug
                if prompt.count("<image>") != len(images):
                    results.append('')
                    continue
                    
                inputs = processor(images=images, text=prompt, return_tensors="pt").to(model.device)
                inputs = inputs.to(torch.bfloat16)
                # try:
                #     inputs = processor(images=images, text=prompt, return_tensors="pt").to(model.device)
                #     inputs = inputs.to(torch.bfloat16)
                # except Exception as e:
                #     results.append('')
                #     print(f"error: {str(e)}")
                #     continue
                
                
                decoded_output = ""
                retry_count = 0
                max_retries = MAX_RETRY
                
                while not decoded_output and retry_count < max_retries:
                    try:
                        
                        output = model.generate(**inputs, max_new_tokens=30, return_dict_in_generate=True, output_hidden_states=True) #TODO is maybe max_new_tokens=30 too low?
                        generated_tokens = output.sequences[:, inputs['input_ids'].shape[-1]:]
                        decoded_output = processor.decode(generated_tokens[0], skip_special_tokens=True)
                        if not decoded_output:
                            retry_count += 1
                            print(f"Retry {retry_count}/{max_retries} for {part_name} due to empty output.")
                            
                    except Exception as e:
                        retry_count += 1
                        print(f"Retry {retry_count}/{max_retries} for {part_name} due to error: {str(e)}")

                if decoded_output:
                    results.append(decoded_output)
                else:
                    results.append('')
                    print(f"Failed to get a non-empty output after {max_retries} retries for {part_name}.")


            save_results_to_file(zip(results, part_data), output_path)
        return output_path

    temp_files = []
    temp_files.append(process_and_save_part(dataset, SETTING))

def main():
    run_and_save()

if __name__ == '__main__':
    main()
