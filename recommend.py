"""
Book KG — Recommendation Service
----------------------------------
Given a seed book title, combines:
  1. Logic layer  — candidates from GraphDB via similarTo / influenceRelevant
  2. Embedding layer — re-ranks candidates by TransE cosine similarity

Usage:
  python recommend.py "The Silmarillion"
  python recommend.py  (uses default seed)
"""

import sys
import torch
import numpy as np
import requests
from rdflib import Graph, Namespace
from rdflib.namespace import RDFS
from pykeen.triples import TriplesFactory
from pykeen.pipeline import PipelineResult

BKG          = Namespace("http://bookkg.org/ontology#")
GRAPHDB_BASE = "http://localhost:7200"
GRAPHDB_REPO = "book_rec"
TOP_N        = 10

# Load labels from KG
print("→ Loading KG labels")
g = Graph()
g.parse("book_kg_enriched.ttl", format="turtle")
labels   = {str(s): str(o) for s, _, o in g.triples((None, RDFS.label, None))}
uri_by_label = {v.lower(): k for k, v in labels.items()}

def get_label(uri): return labels.get(str(uri), str(uri).split("/")[-1])

# Load trained TransE model 
print("→ Loading TransE model...")
tf_train     = TriplesFactory.from_path_binary("models/transe/training_triples")
model        = torch.load("models/transe/trained_model.pkl", map_location="cpu")
model.eval()
entity_to_id = tf_train.entity_to_id

#Step 1: Get seed book URI
seed_title = sys.argv[1] if len(sys.argv) > 1 else "The Silmarillion"
seed_uri   = uri_by_label.get(seed_title.lower())

if seed_uri is None:
    # fuzzy match — find closest title
    matches = [(t, u) for t, u in uri_by_label.items() if seed_title.lower() in t]
    if matches:
        seed_uri = matches[0][1]
        seed_title = get_label(seed_uri)
        print(f"  Matched to: '{seed_title}'")
    else:
        print(f"  ✗ Book '{seed_title}' not found in KG.")
        sys.exit(1)

print(f"\n Seed book: {seed_title}")

# Step 2: Logic layer — SPARQL query on GraphDB
print("\n→ Querying GraphDB for logic-based candidates...")

SPARQL_QUERY = f"""
PREFIX bkg:  <http://bookkg.org/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?candidate ?label ?connection
WHERE {{
  BIND(<{seed_uri}> AS ?seed)

  {{
    # Rule 1/5: directly similar via shared theme or author
    ?seed bkg:similarTo ?candidate .
    BIND("similarTo" AS ?connection)
  }} UNION {{
    # Rule 2: influence-relevant books
    ?seedAuthor bkg:wrote ?seed .
    ?seedAuthor bkg:influenceRelevant ?candidate .
    BIND("influenceRelevant" AS ?connection)
  }} UNION {{
    # Transitive: author transitively influenced another author who wrote candidate
    ?seedAuthor bkg:wrote ?seed .
    ?seedAuthor bkg:transitiveInfluence ?otherAuthor .
    ?otherAuthor bkg:wrote ?candidate .
    BIND("transitiveInfluence" AS ?connection)
  }}

  ?candidate a bkg:Book .
  ?candidate rdfs:label ?label .
  FILTER(?candidate != <{seed_uri}>)
}}
LIMIT 200
"""

r = requests.post(
    f"{GRAPHDB_BASE}/repositories/{GRAPHDB_REPO}",
    headers={"Accept": "application/sparql-results+json",
             "Content-Type": "application/x-www-form-urlencoded"},
    data={"query": SPARQL_QUERY}
)

if r.status_code != 200:
    print(f"  ✗ GraphDB error: {r.status_code}")
    sys.exit(1)

bindings = r.json()["results"]["bindings"]
candidates = {
    b["candidate"]["value"]: {
        "label":      b["label"]["value"],
        "connection": b["connection"]["value"]
    }
    for b in bindings
}
print(f"  ✓ {len(candidates)} logic-based candidates found")

# Step 3: Embedding layer — re-rank by cosine similarity
print("→ Re-ranking by embedding similarity...")

if seed_uri not in entity_to_id:
    print(f"  ✗ Seed book not in embedding model.")
    sys.exit(1)

seed_id  = entity_to_id[seed_uri]
seed_vec = model.entity_representations[0](
    torch.tensor([seed_id])
).detach().numpy().flatten()

scored = []
for uri, info in candidates.items():
    if uri not in entity_to_id:
        continue
    b_id  = entity_to_id[uri]
    b_vec = model.entity_representations[0](
        torch.tensor([b_id])
    ).detach().numpy().flatten()
    sim = float(np.dot(seed_vec, b_vec) /
                (np.linalg.norm(seed_vec) * np.linalg.norm(b_vec) + 1e-9))
    scored.append((uri, info["label"], info["connection"], sim))

scored.sort(key=lambda x: x[3], reverse=True)

# Step 4: Print recommendations
print(f"\n Top-{TOP_N} recommendations for '{seed_title}':\n")
print(f"  {'#':<4} {'Title':<45} {'Connection':<22} {'Sim':>6}")
print(f"  {'-'*4} {'-'*45} {'-'*22} {'-'*6}")
for i, (uri, label, conn, sim) in enumerate(scored[:TOP_N], 1):
    print(f"  {i:<4} {label:<45} {conn:<22} {sim:>6.4f}")

# Step 5: Show seed book themes for context
print(f"\nThemes of '{seed_title}':")
seed_themes = [get_label(str(o)) for s, p, o in g.triples((None, BKG.hasTopic, None))
               if str(s) == seed_uri]
print(f"  {', '.join(seed_themes[:10])}")

print("\nRecommendation complete.")
