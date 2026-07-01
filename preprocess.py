import pandas as pd
from datasets import load_dataset

def preprocess_bengali_dataset():
    dataset = load_dataset("barunsaha/bangla_sahitya", split="train")
    df = pd.DataFrame(dataset)
    
    df_prose = df[df['tag'] == 'prose'].copy()
    
    processed_chunks = []
    
    for _, row in df_prose.iterrows():
        author = row['author']
        title = row['title']
        text = row['text']
        
        paragraphs = text.split('\n')
        
        current_chunk = []
        current_word_count = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            words = para.split()
            current_chunk.append(para)
            current_word_count += len(words)
            
            if current_word_count >= 500:
                full_chunk_text = "\n".join(current_chunk)
                processed_chunks.append({
                    "author": author,
                    "title": title,
                    "bengali_narrative_chunk": full_chunk_text
                })
                current_chunk = []
                current_word_count = 0
 
        if current_chunk:
            processed_chunks.append({
                "author": author,
                "title": title,
                "bengali_narrative_chunk": "\n".join(current_chunk)
            })

    output_df = pd.DataFrame(processed_chunks)

    output_file = "bengali_comic_source_ready.csv"
    output_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Success")

if __name__ == "__main__":
    preprocess_bengali_dataset()