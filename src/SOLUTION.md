# Day 7 solution

This implementation provides TF-IDF, dense cosine, and hybrid RRF retrieval,
an OpenAI-compatible model/embedding adapter, an evidence-grounded agent loop,
traces, and retrieval metrics. It uses only the Python standard library.

## Configure

Copy `.env.example` to `.env` and fill real provider values. Never commit
`.env`. Dense document vectors and query embeddings must use the same model.
For a local OpenAI-compatible endpoint such as Ollama, use a localhost base URL;
the client does not require an API key for localhost.

## Run

```powershell
python src\run_lab.py index --chunks data\chunks.json --output data\vectors.json

python src\run_lab.py benchmark --strategy tfidf --chunks data\chunks.json --questions data\questions.json --split dev --output results\dev-tfidf
python src\run_lab.py benchmark --strategy dense --chunks data\chunks.json --questions data\questions.json --vectors data\vectors.json --split dev --output results\dev-dense
python src\run_lab.py benchmark --strategy hybrid --chunks data\chunks.json --questions data\questions.json --vectors data\vectors.json --split dev --output results\dev-hybrid

python src\run_lab.py agent --strategy hybrid --chunks data\chunks.json --vectors data\vectors.json --cases data\agent_cases.json --split dev --output results\agent-dev-traces.json
```

Tune only on dev. Freeze the selected arguments before changing the split to
test. After manually reviewing answers and citation support, aggregate the
labels with `starter/agent_metrics.py`.
