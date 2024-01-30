Consider the recent use of _direct preference optimization_ (DPO) with open source tools such as `Argilla` and `Distilabel` to identify and fix data quality issues in the `Zephyr-7B-beta` dataset. This resulted in the `Notus-7B-v1` model, which was created by a relatively small R&D team -- "GPU-poor" -- and then gained high ranking on the Hugging Face leaderboards.

  - <https://huggingface.co/argilla/notus-7b-v1>
  - <https://argilla.io/blog/notus7b/>

Andrew Ng:
<https://www.linkedin.com/posts/andrewyng_ai-discovers-new-antibiotics-openai-revamps-activity-7151282706947969025-WV2v>

> While it's always nice to have massive numbers of NVIDIA H100 or AMD MI300X GPUs, this work is another illustration — out of many, I want to emphasize — that deep thinking with only modest computational resources can carry you far.

"Direct Preference Optimization: Your Language Model is Secretly a Reward Model"  
Rafael Rafailov, et al.  
<https://arxiv.org/abs/2305.18290>

RE projects in particular tend to use Wikidata _labels_ (not IRIs) to train models; these are descriptive but not computable

Components such as NER and RE could be enhanced by reworking the data quality for training data, benchmarks, evals, etc.

  - `SpanMarker` provides a framework for iteration on NER, to fine-tune for specific KGs

  - `OpenNRE` provides a framework for iteration on RE, to fine-tune for specific KGs

Data-first iterations on these components can take advantage of DPO, sparse fine-tuning, pruning, quantization, and so on, while the _lemma graph_ plus its topological transforms provide enhanced tokenization and better context for training.
