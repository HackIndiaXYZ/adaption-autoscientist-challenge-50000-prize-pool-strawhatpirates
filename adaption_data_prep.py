import os
import json
import time
import re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests
from tqdm import tqdm

# API Gateway Configuration
KRUTRIM_URL = "https://cloud.olakrutrim.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-oss-20b"

# Resilient Backoff Parameters
MAX_RETRIES = 5
INITIAL_BACKOFF = 4.0  

def parse_arguments():
    parser = argparse.ArgumentParser(description="Clean Bengali Dataset and Batch-Generate English Comic Scripts via Krutrim.")
    parser.add_argument("--api-key", type=str, default=None, help="Your Krutrim Cloud API Key")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Krutrim model string to use")
    parser.add_argument("--limit-rows", type=int, default=None, help="Limit rows to process (useful for dry runs / testing)")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel request workers")
    return parser.parse_args()

def clean_and_load_dataset(filepath):
    print(f"Loading preprocessed dataset: {filepath}")
    df = pd.read_csv(filepath)
    initial_count = len(df)
    
    # Filter out long rows (>800 words) to prevent OOMs
    print("Filtering out outlier chunks containing over 800 words...")
    df['word_count'] = df['bengali_narrative_chunk'].apply(lambda x: len(str(x).split()))
    df = df[df['word_count'] < 800].copy()
    filtered_count = len(df)
    print(f"Filtered {initial_count - filtered_count} rows. Retained {filtered_count} optimal rows.")
    
    # Safe imputing for empty/missing title rows
    print("Imputing missing title fields with localized defaults...")
    df['title'] = df['title'].fillna("Bengali Literature Segment")
    
    return df

def load_checkpoints(checkpoint_path):
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                print(f"Checkpoint found! Resuming progress from {len(checkpoint_data)} completed rows.")
                return checkpoint_data
        except Exception as e:
            print(f"Error reading checkpoint file ({e}). Starting fresh.")
    return {}

def get_system_instructions():
    return (
        "You are an expert bilingual comic book screenwriter and cultural translator.\n"
        "Your task is to analyze raw Bengali literary narrative chunks and translate their conceptual, "
        "dramatic, and emotional context into a structured, cinematic English graphic novel script.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "- DO NOT output a 'Thinking Process'.\n"
        "- DO NOT output reasoning or explanations.\n"
        "- Output ONLY a raw, valid JSON object.\n"
        "- Start your response immediately with the { character.\n\n"
        "JSON SCHEMA STRUCTURE:\n"
        "{\n"
        "  \"art_style\": \"Universal visual style matching the mood (e.g. Gritty Noir, 19th-century watercolor illustration, vintage ink comic sketch)\",\n"
        "  \"panels\": [\n"
        "    {\n"
        "      \"panel\": 1,\n"
        "      \"setting\": \"Detailed visual description of scenery, actors, and lighting prompts suitable for image generators.\",\n"
        "      \"caption\": \"A narrative caption or scene voiceover explaining the progression.\",\n"
        "      \"dialogue\": \"Exact character speech or dialogue (leave empty string if purely narrative action).\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Constraint: Generate between 1 and 3 panels maximum."
    )

def generate_row_target(row, api_key, model_name):
    bengali_text = row['bengali_narrative_chunk']
    meta_context = f"Author: {row['author']} | Title: {row['title']}"
    
    # Build unified instructions payload matching the multimodal specification precisely
    combined_prompt = (
        f"{get_system_instructions()}\n\n"
        f"--- CONTEXT ---\n"
        f"{meta_context}\n\n"
        f"--- BENGALI TEXT TO ADAPT ---\n"
        f"{bengali_text}"
    )
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": combined_prompt 
            }
        ],
        "temperature": 0.1, 
        "max_tokens": 1000 
    }
    
    backoff = INITIAL_BACKOFF
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(KRUTRIM_URL, headers=headers, json=payload, timeout=120)
            
            if response.status_code == 200:
                res_data = response.json()
                if 'choices' in res_data and len(res_data['choices']) > 0:
                    raw_text = res_data['choices'][0]['message']['content']
                    if raw_text:
                        return raw_text.strip()
                    else:
                        print(f"\n Content is None for {row['title']}. Finish reason: {res_data['choices'][0].get('finish_reason')}")
                        if res_data['choices'][0].get('finish_reason') == 'length':
                            return res_data['choices'][0]['message'].get('content', '')
                print(f"\n Unexpected response structure for row {row['title']}: {res_data}")
                return None
            elif response.status_code == 401:
                print(f"\n Auth Error 401: API Key expired or invalid.")
                return None
            elif response.status_code >= 500:
                print(f"\n Gateway 500 Error for row {row['title']} (Attempt {attempt+1}/{MAX_RETRIES}). Server overloaded or context too large. Cooling down for {backoff}s...")
                time.sleep(backoff)
                backoff *= 1.5
            else:
                print(f"\n HTTP Error {response.status_code} for row {row['title']}: {response.text[:200]}")
                time.sleep(backoff)
                backoff *= 1.5
                
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"\n Connection Failure on row: {row['title']} | Error: {e}")
                return None
            print(f"\n Network hitch for row {row['title']}. Retrying in {backoff}s...")
            time.sleep(backoff)
            backoff *= 1.5

def extract_valid_json(raw_text):
    if not raw_text:
        return None
    
    cleaned = raw_text.strip()
    
    # 1. Try direct parsing first
    try:
        json_obj = json.loads(cleaned)
        if 'art_style' in json_obj and 'panels' in json_obj:
            return cleaned
    except json.JSONDecodeError:
        pass

    # 2. Aggressive extraction: Find everything between the first { and last }
    try:
        start_idx = cleaned.find('{')
        end_idx = cleaned.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            extracted_json_str = cleaned[start_idx:end_idx + 1]
            
            json_obj = json.loads(extracted_json_str)
            if 'art_style' in json_obj and 'panels' in json_obj:
                return extracted_json_str
    except Exception as e:
        with open("failed_json_debug.log", "a", encoding="utf-8") as debug_file:
            debug_file.write(f"\n--- FAILED JSON EXTRACTION ---\n")
            debug_file.write(f"Raw Text:\n{raw_text}\n")
            debug_file.write(f"Exception: {e}\n")
            debug_file.write(f"------------------------------\n")
        pass
        
    return None

def main():
    args = parse_arguments()
    
    api_key = args.api_key or os.environ.get("KRUTRIM_CLOUD_API_KEY")
    if not api_key:
        print("Error: Krutrim Cloud API key is missing!")
        print("Please export your API key: export KRUTRIM_CLOUD_API_KEY='your_key'")
        return
        
    df = clean_and_load_dataset("bengali_comic_source_ready.csv")
    
    if args.limit_rows:
        df = df.head(args.limit_rows)
        print(f"Limit applied: Processing only the first {args.limit_rows} rows.")
        
    checkpoint_path = "comic_dataset_checkpoint.json"
    completed_rows = load_checkpoints(checkpoint_path)
    
    pending_rows = []
    for idx, row in df.iterrows():
        row_id = f"{row['title']}_{idx}"
        if row_id not in completed_rows:
            pending_rows.append((row_id, row))
            
    if not pending_rows:
        print("All rows completed! Packaging final output configuration.")
        build_final_dataset(completed_rows)
        return

    print(f"Total pending samples to translate: {len(pending_rows)}")
    print(f"Running pipeline with {args.workers} parallel workers using: {args.model}")
    print("Tracking live progress below...")

    checkpoint_counter = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_id = {
            executor.submit(generate_row_target, row, api_key, args.model): (row_id, row) 
            for row_id, row in pending_rows
        }
        
        for future in tqdm(as_completed(future_to_id), total=len(future_to_id), desc="Labeling Dataset"):
            row_id, row = future_to_id[future]
            try:
                raw_response = future.result()
                valid_json_str = extract_valid_json(raw_response)
                
                if valid_json_str:
                    completed_rows[row_id] = {
                        "bengali_narrative_chunk": row["bengali_narrative_chunk"],
                        "target_comic_json": valid_json_str
                    }
                    checkpoint_counter += 1
                else:
                    print(f"\n Skipping row due to JSON formatting failures: {row_id}")
            except Exception as exc:
                print(f"\n Worker exception on row {row_id}: {exc}")
                
            if checkpoint_counter >= 10:
                save_checkpoint(completed_rows, checkpoint_path)
                checkpoint_counter = 0

    save_checkpoint(completed_rows, checkpoint_path)
    build_final_dataset(completed_rows)

def save_checkpoint(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def build_final_dataset(completed_rows):
    print("Packing into final Adaption Instruction Dataset layout...")
    final_dataset = []
    for row_id, data in completed_rows.items():
        instruction_pair = {
            "instruction": (
                "Convert the following Bengali historical narrative segment into a structured "
                "English comic book page layout consisting of individual panels with precise setting directions, "
                "captions, and narrative dialogues."
            ),
            "input": data["bengali_narrative_chunk"],
            "output": data["target_comic_json"]
        }
        final_dataset.append(instruction_pair)
        
    output_filepath = "bengali_to_comic_adaptive_train.json"
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, indent=2, ensure_ascii=False)
        
    print(f"\n Process Complete! Cleaned and generated target outputs for {len(final_dataset)} rows.")
    print(f"Final Target Dataset saved at: {output_filepath}")
    print("Ready to upload directly to Adaption's platform under Step 3!")

if __name__ == "__main__":
    main()