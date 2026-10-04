# Natural Language Processing Assignments

Assignments from my Natural Language Processing course, covering tokenization and statistical language models, text classification, word embeddings, Transformer fine-tuning, and document retrieval.

Most of the work is in Jupyter notebooks, alongside the assignment instructions, explanations, experiments, and saved outputs. The notebooks include both Persian and English text.

## Assignments

| Assignment | Topics |
| --- | --- |
| [CA1 — Tokenization and language modeling](CA1/NLP_CA1_Mirshabani_810801080.ipynb) | Regex, edit-distance spelling correction, BPE and WordPiece tokenizers, and Persian n-gram models with smoothing and temperature sampling. |
| [CA2 — Text classification](CA2/NLP_CA2_Mirshabani_810801080.ipynb) | Manual implementations of logistic regression and multinomial Naive Bayes for spam detection, followed by phishing-URL classification with different feature sets. |
| [CA3 — Word embeddings and neural classification](CA3/NLP_CA3_Mirshabani_810801080.ipynb) | CBOW and Skip-Gram models in PyTorch, embedding similarity and visualization, and AG News classification using pretrained FastText vectors and an MLP. |
| [CA4 Q1 — Text-to-SQL](CA4/Q1/NLP_CA4_Q1_Mirshabani_810801080.ipynb) | Fine-tuning BART and GPT-2 to generate SQL from questions and database schemas, with exact-match evaluation and error analysis. |
| [CA4 Q2 — Instruction following](CA4/templates/NLP_CA4_Q2_Template.ipynb) | Assignment template for LoRA fine-tuning and IFEval evaluation; the implementation is not included. |
| [CA5 Q1 — Legal-document retrieval](CA5/Q1/NLP_CA5_Q1_Mirshabani_810801080.ipynb) | Partial work on Persian PDF extraction, DeepSeek OCR, article chunking, embeddings, and a retrieval agent. |
| [CA5 Q2 — Travel assistant](CA5/Q2/NLP_CA5_Q2_Mirshabani_810801080.ipynb) | Initial setup for a LangGraph travel assistant; most of the implementation is still unfinished. |

## Finding your way around

Each `CA` folder contains its notebooks and related data. Original assignment prompts are kept in `templates/`, while workshop notebooks and recordings are under `workshops/`. Drafts and small experiments are kept alongside the assignments or in `scratch/`.

The main libraries used across the completed work are NumPy, pandas, scikit-learn, PyTorch, Hugging Face Datasets and Transformers, and FastText.

## Running the notebooks

Open a notebook in Jupyter, VS Code, or Google Colab and check its setup cells before running it. For notebooks that use local data, use the notebook's directory as the working directory. Some later notebooks use Colab paths, Google Drive, a GPU, or external APIs and need the corresponding setup.

The large FastText model used in CA3 is excluded from Git. Its download commands are included in the notebook. Pretrained models and some datasets also need to be downloaded when running the assignments.
