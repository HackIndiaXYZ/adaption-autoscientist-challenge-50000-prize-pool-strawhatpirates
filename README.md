# KathaChitra AI: Bengali Literature to Comic Strip Translator

This repository contains the codebase and data pipelines developed by team "StrawHatPirates" for the Adaption AutoScientist Challenge.

This model is a fine-tuned LoRA adapter based on `meta-llama/Llama-3.2-3B-Instruct`, trained autonomously using the Adaption AI AutoScientist framework. It specializes in translating complex, historical Bengali literary prose into visually vivid, structured English comic book page layouts formatted cleanly in JSON.

---

## Repository Structure

*   `preprocess.py`: Extracts and prepares raw Bengali prose from datasets.
*   `adaption_data_prep.py`: Prepares the dataset via LLM-assisted cultural translation and comic structuring.
*   `bengali_comic_source_ready.csv`: Preprocessed dataset containing author, title, and Bengali narrative chunks.
*   `bengali_to_comic_adaptive_train.json`: The final instruction dataset format containing instruction, input, and JSON target output.

---

## Stepping Stones to Model Development

Developing this translation and comic-script generation model follows a structured four-stage process:

### 1. Data Processing and Chunking (`preprocess.py`)
Historical Bengali literature often contains long narrative chunks that need splitting to fit context windows.
*   **Action**: Run `python preprocess.py`
*   **Details**: Loads the `barunsaha/bangla_sahitya` dataset, filters for prose, and groups paragraphs into narrative chunks of approximately 500 words.
*   **Output**: Saves the result to `bengali_comic_source_ready.csv` containing `author`, `title`, and `bengali_narrative_chunk`.

### 2. LLM-Assisted Cultural Translation and Structuring (`adaption_data_prep.py`)
To train the adapter, we need a high-quality parallel dataset of Bengali prose mapped to structured English comic layouts.
*   **Action**: Run `python adaption_data_prep.py --api-key YOUR_API_KEY`
*   **Details**:
    *   Filters out outlier chunks containing over 800 words to ensure optimal context lengths.
    *   Uses Krutrim Cloud API models to perform cultural translation and panels generation.
    *   Asks the LLM to output a strict JSON layout including:
        *   `art_style`: Visual style matching the mood.
        *   `panels`: An array containing `panel`, `setting` (visual description for image generators), `caption` (narration/voiceover), and `dialogue` (exact character speech).
    *   Employs checkpointing and thread pools for parallel requests and resilience.
*   **Output**: Generates `bengali_to_comic_adaptive_train.json` formatted as an instruction tuning dataset.

### 3. Autonomous Training (Adaption AI AutoScientist)
The structured JSON dataset is loaded onto the Adaption AI platform.
*   **Action**: The dataset `bengali_to_comic_adaptive_train.json` is used by the AutoScientist framework to train the model.
*   **Details**: The framework autonomously fine-tunes `meta-llama/Llama-3.2-3B-Instruct` using Low-Rank Adaptation (LoRA).
*   **Output**: Model configs, safetensors are outputted and accessible on Adaption's web platform, downloaded and published on HF and Kaggle

### Important links
* Kaggle model: https://www.kaggle.com/models/sarthakdas89/bengali-narrative-to-comic-model
* Kaggle dataset: https://www.kaggle.com/datasets/sarthakdas89/bengali-narrative-to-comic
* Kaggle notebook: https://www.kaggle.com/code/sarthakdas89/kathachitra
* HuggingFace model: https://huggingface.co/sarthakd57/bengali_narrative_to_comic_model
* HuggingFace space(live demo): https://huggingface.co/spaces/sarthakd57/KathaChitra
* HuggingFace adapted dataset: https://huggingface.co/datasets/sarthakd57/bengali_narrative_to_comic
* BanglaSahitya dataset: https://huggingface.co/datasets/barunsaha/bangla_sahitya
* Linkedin post: https://www.linkedin.com/posts/sarthak-das-63014521b_greetings-as-participant-of-adaptions-ugcPost-7478001574334668800-mwbP/?utm_source=share&utm_medium=member_desktop&rcm=ACoAADdfJUQBbuKE86-IrmKpmiIame6y4oXzHCc
* X(formerly twitter) post: https://x.com/saddyk55/status/2072241116607594927?s=20




