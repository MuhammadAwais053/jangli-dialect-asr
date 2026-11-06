# Jangli speech recognition with Whisper

For my computer science thesis, I fine-tuned `openai/whisper-small` to transcribe Jangli speech from the Sahiwal and Sargodha districts of Punjab, Pakistan. On the 218 test clips used in the original experiment, fine-tuning reduced word error rate (WER) from **108.66% to 86.09%**, a decrease of **22.57 percentage points**. Transcription errors remain frequent.

The [fine-tuned model](https://huggingface.co/awais064/whisper-jangli-small) is available on Hugging Face. This repository contains the [training notebook](notebook/Jangli_Finetune_V1.ipynb), a sample of the clip metadata, and examples of the model’s predictions. The original recordings are private to protect the speakers.

## Try the model on a short recording

The example below uses your own audio file and the published model weights. Use a WAV file containing up to 30 seconds of speech and save it as `sample.wav`. The processor comes from the original Whisper Small checkpoint. The example uses the Urdu language setting and the transcription task used in the experiment.

```bash
python -m pip install torch transformers librosa soundfile
```

```python
import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained(
    "awais064/whisper-jangli-small"
).to(device)
model.eval()

audio, sample_rate = librosa.load("sample.wav", sr=16000, mono=True)
if len(audio) > 30 * sample_rate:
    raise ValueError("Choose a clip of 30 seconds or less for this example.")

features = processor(
    audio,
    sampling_rate=sample_rate,
    return_tensors="pt",
).input_features.to(device)

with torch.inference_mode():
    token_ids = model.generate(
        features,
        language="urdu",
        task="transcribe",
        max_length=225,
    )

print(processor.batch_decode(token_ids, skip_special_tokens=True)[0])
```

## Recordings and transcripts

I cleaned the source recordings in Audacity and used Whisper to produce first-pass transcripts. During manual correction, I kept a spelling log for dialect words. The extent of review varied across the dataset, and some drafts received only partial correction. The number of test transcripts checked in full against the audio is undocumented.

The notebook pairs each cleaned recording with its timestamped labels, converts the audio to 16 kHz mono, and exports one WAV clip per labeled segment. This produced **2,179 clips from 93 source recordings**. The [metadata sample](data/metadata_sample.csv) contains 30 rows and shows the clip filename, transcript, duration in seconds, and source recording.

| Split | Clips |
|---|---:|
| Training | 1,743 |
| Validation | 218 |
| Test | 218 |
| **Total** | **2,179** |

The notebook shuffled individual clips with a fixed seed of `42` before making the 80/10/10 split. Clips from the same source recording could therefore fall into different splits. The scores below describe performance on held-out **clips**. Performance on entirely new speakers or recordings remains unmeasured.

## Training and evaluation

I fine-tuned Whisper Small for four epochs on a Google Colab T4 GPU, using a learning rate of `1e-5`, a per-device training batch size of `8`, and two gradient accumulation steps. The notebook uses validation WER to select the best saved checkpoint. It then evaluates the pretrained baseline and the fine-tuned model on the same 218 test clips.

| Model | Test WER |
|---|---:|
| Whisper Small before fine-tuning | 108.66% |
| Whisper Small after fine-tuning | 86.09% |

WER counts word substitutions, deletions, and insertions, then divides their sum by the number of words in the reference transcripts. Insertions can push the score above 100%. The evaluation cell passes the decoded predictions and references directly to the WER metric, so spelling and word spacing can affect the result.

One [published example](results/sample_predictions.csv) shows the kind of difference this scoring captures. The final word is `صونی` in the reference and `سونی` in the model output. WER counts that as one substitution. The CSV contains ten examples for inspection. The [saved result summary](results/results_summary.txt) reports scores for the full test split.

## Using the training notebook

The [Colab notebook](notebook/Jangli_Finetune_V1.ipynb) includes clip creation, dataset splitting, feature preparation, fine-tuning, and evaluation. To adapt it to your own recordings:

1. Prepare cleaned WAV recordings and matching UTF-8 label files. A recording and its label file must share the same base filename. Each label row contains a start time in seconds, an end time in seconds, and the transcript, separated by tabs.
2. Update the Google Drive paths in the notebook to point to your audio, label, and output folders. The original project path appears in several cells.
3. Use the [dependency list](requirements.txt) when setting up the environment. Package versions are unpinned. The notebook’s training and evaluation cells assume a CUDA GPU.
4. Run clip preparation and inspect the generated `metadata.csv` before training. If your aim is to evaluate new recordings or new speakers, change the split step to group clips by recording or speaker before running the experiment.

The public metadata and prediction samples let readers inspect the data format and some transcription errors. Reproducing the original training run or independently recomputing its full test WER requires access to the private dataset.
