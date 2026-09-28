# Knowledge Graph-Based Book Recommendations
**Margarida Gonçalves — 12142151**  
Knowledge Graphs (192.194) — TU Wien, SS 2026

---

## Project Structure

```
├── 2 - construction/
│   ├── ingestion.py         # Wikidata SPARQL + OpenLibrary API pipeline
│   ├── conversion.py        # Converts CSVs to RDF Turtle (book_kg.ttl)
├── 3 - ML/
│   ├── embeddings.py        # TransE training and evaluation via PyKEEN        
├── 4 - logic/
│   ├── reasoning.py         # Datalog rules via SPARQL CONSTRUCT (RDFLib)
├── recommend.py             # Recommendation service (logic + embeddings)
├── requirements.txt     
└── README.md                 
```

---

## Setup

### Requirements
- Python 3.11+
- GraphDB (free version): https://graphdb.ontotext.com

### Install dependencies
```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

---

## Running the Pipeline

### Step 1 — Data Ingestion
```bash
python 2-construction\ingestion.py
```
Produces: `triples_wrote.csv`, `triples_influenced_by.csv`, `triples_has_genre.csv`,  
`triples_has_topic.csv`, `entities_books.csv`, `entities_authors.csv`

Then convert to RDF:
```bash
python 2-construction\conversion.py 
```
Produces: `book_kg.ttl`

Load `book_kg.ttl` into GraphDB:
- Open GraphDB → Create repository named `book_rec`
- Import → RDF Files → Upload `book_kg.ttl`

### Step 2 — Logical Reasoning
```bash
python 4-logic\reasoning.py
```
Loads inferred triples into GraphDB automatically.

### Step 3 — KG Embeddings
```bash
python 3-ML\embeddings.py
```

### Step 4 — Recommendations
```bash
python recommend.py "The Invention of Hugo Cabret"
python recommend.py "The Silmarillion"
```
Requires: GraphDB running, `book_kg_enriched.ttl` present, `models/transe/` trained.

---

## Data Sources
- **Wikidata**: https://www.wikidata.org (SPARQL endpoint, open license CC0)
- **OpenLibrary**: https://openlibrary.org (REST API, open license CC0)

---
