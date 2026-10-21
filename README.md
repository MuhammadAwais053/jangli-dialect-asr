# Fine-Tuning Whisper for a Low-Resource Dialect (Jangli Dialect ASR)

Fine-tuning OpenAI's Whisper speech-recognition model to transcribe **Jangli**,
a spoken dialect from the Sahiwal and Sargodha districts of Punjab, Pakistan,
with no prior digital transcription resources. Built as the technical
foundation of a university Computer Science thesis.

## Results

| | Word Error Rate |
|---|---|
| Baseline (pre-fine-tuning) | 108.66% |
| Fine-tuned | 86.09% |
| **Improvement** | **22.57 percentage points** |

See [`results/results_summary.txt`](results/results_summary.txt) and
[`results/sample_predictions.csv`](results/sample_predictions.csv) for details.

## Model

The fine-tuned model is hosted on Hugging Face Hub:
**[awais064/jangli-whisper-small](https://huggingface.co/awais064/whisper-jangli-small)**

## Approach

1. **Data collection** — audio recordings of Jangli speakers across Sahiwal and Sargodha.
2. **Cleaning** — noise reduction and normalization (Audacity), standardized to 16kHz mono.
3. **Draft transcription** — OpenAI Whisper generated first-pass transcripts (via Audacity's
   OpenVINO plugin and Google Colab), since no existing tool understands Jangli.
4. **Manual correction** — every draft was corrected by hand against the source audio,
   with a maintained spelling-consistency log for dialect-specific vocabulary.
5. **Dataset construction** — corrected transcripts were time-aligned and cut into
   clips, producing a structured dataset (80/10/10 train/validation/test split).
   A representative sample is in [`data/metadata_sample.csv`](data/metadata_sample.csv).
6. **Fine-tuning** — `openai/whisper-small` fine-tuned using Hugging Face Transformers
   and PyTorch on Google Colab (T4 GPU), 4 epochs.
7. **Evaluation** — Word Error Rate measured on a held-out test set, comparing the
   baseline (untrained) model against the fine-tuned result.


Note: the full audio dataset is not published here, to protect the privacy of
the recorded speakers. The notebook and sample files are provided to document
the methodology and demonstrate reproducibility.

## Tools Used

OpenAI Whisper · Hugging Face Transformers · PyTorch · Google Colab ·
Audacity + OpenVINO AI Plugins · Python (pandas, pydub)

## Limitations & Future Work

- Dataset size (~150-170 minutes) is small by ASR fine-tuning standards;
  more annotated data would likely improve results further.
- A portion of the dataset relied on AI-generated drafts with partial manual
  correction, due to project timeline constraints.
- The original project scope proposed meta-learning (MAML) and contrastive
  learning approaches for dialect differentiation; these were descoped in
  favor of standard fine-tuning given dataset size, and remain a direction
  for future work with a larger, multi-dialect dataset.
