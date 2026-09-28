"""
Book KG — Reasoning Pipeline (RDFLib)
--------------------------------------
Applies Datalog-style rules via SPARQL CONSTRUCT queries in RDFLib.
Inferred triples are added back into the graph iteratively until
no new triples are produced (fixpoint).

Rules implemented:
  Rule 1 — Direct theme similarity (similarTo)
  Rule 2 — Influence-based book relevance (influenceRelevant)
  Rule 3 — Transitive influence (RECURSIVE, fixpoint)
  Rule 4 — Author similarity via shared influence (similarAuthor)
  Rule 5 — Cross-author book similarity (similarTo via similarAuthor)

Output:
  inferred_triples.ttl  — only the inferred triples
  book_kg_enriched.ttl  — original + inferred triples
"""

import requests
from rdflib import Graph, Namespace
from rdflib.namespace import RDF, RDFS, OWL

BKG  = Namespace("http://bookkg.org/ontology#")
BOOK = Namespace("http://bookkg.org/book/")
AUTH = Namespace("http://bookkg.org/author/")

GRAPHDB_BASE = "http://localhost:7200"
GRAPHDB_REPO = "book_rec"

# Load KG

print("→ Loading book_kg.ttl...")
g = Graph()
g.parse("book_kg.ttl", format="turtle")
print(f"  ✓ {len(g)} triples loaded")

# SPARQL CONSTRUCT rules

PREFIXES = """
PREFIX bkg:  <http://bookkg.org/ontology#>
PREFIX book: <http://bookkg.org/book/>
PREFIX auth: <http://bookkg.org/author/>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""

RULES = {

    # Rule 1: two books sharing a theme are similar
    "Rule 1 — Direct theme similarity": """
        CONSTRUCT { ?b1 bkg:similarTo ?b2 }
        WHERE {
            ?b1 bkg:hasTopic ?t .
            ?b2 bkg:hasTopic ?t .
            ?b1 a bkg:Book .
            ?b2 a bkg:Book .
            FILTER(?b1 != ?b2)
        }
    """,

    # Rule 2: influence-based book relevance
    "Rule 2 — Influence-based relevance": """
        CONSTRUCT { ?a bkg:influenceRelevant ?x }
        WHERE {
            ?a bkg:influencedBy ?b .
            ?b bkg:wrote ?x .
            ?x a bkg:Book .
        }
    """,

    # Rule 3a: base case for transitive influence
    "Rule 3a — Transitive influence (base)": """
        CONSTRUCT { ?a bkg:transitiveInfluence ?b }
        WHERE {
            ?a bkg:influencedBy ?b .
        }
    """,

    # Rule 4: authors influenced by the same author are similar
    "Rule 4 — Similar authors": """
        CONSTRUCT { ?a1 bkg:similarAuthor ?a2 }
        WHERE {
            ?a1 bkg:influencedBy ?x .
            ?a2 bkg:influencedBy ?x .
            ?a1 a bkg:Author .
            ?a2 a bkg:Author .
            FILTER(?a1 != ?a2)
        }
    """,

    # Rule 5: books by similar authors are similar
    "Rule 5 — Cross-author book similarity": """
        CONSTRUCT { ?b1 bkg:similarTo ?b2 }
        WHERE {
            ?a1 bkg:similarAuthor ?a2 .
            ?a1 bkg:wrote ?b1 .
            ?a2 bkg:wrote ?b2 .
            ?b1 a bkg:Book .
            ?b2 a bkg:Book .
            FILTER(?b1 != ?b2)
        }
    """,
}

# Rule 3b is recursive — handled separately below
RULE_3B = """
    CONSTRUCT { ?a bkg:transitiveInfluence ?c }
    WHERE {
        ?a bkg:influencedBy ?b .
        ?b bkg:transitiveInfluence ?c .
    }
"""

# Apply rules

def apply_rule(graph, rule_sparql):
    new_triples = graph.query(PREFIXES + rule_sparql)
    before = len(graph)
    for triple in new_triples:
        graph.add(triple)
    return len(graph) - before

print("\n→ Applying rules...")

for name, rule in RULES.items():
    added = apply_rule(g, rule)
    print(f"  {name}: +{added} triples")

# Rule 3b to fixpoint
print("  Rule 3b — Transitive influence (recursive, fixpoint):")
iteration = 0
while True:
    added = apply_rule(g, RULE_3B)
    iteration += 1
    print(f"    iteration {iteration}: +{added} triples")
    if added == 0:
        break

# Re-apply Rule 5 after Rule 4
added = apply_rule(g, RULES["Rule 5 — Cross-author book similarity"])
print(f"  Rule 5 (re-run after Rule 4): +{added} triples")

print(f"\n  ✓ Total triples after reasoning: {len(g)}")

# Count inferred predicates

print("\n→ Inferred triple counts:")
for pred, label in [
    ("bkg:similarTo",           "similarTo"),
    ("bkg:influenceRelevant",   "influenceRelevant"),
    ("bkg:transitiveInfluence", "transitiveInfluence"),
    ("bkg:similarAuthor",       "similarAuthor"),
]:
    q = PREFIXES + f"SELECT (COUNT(*) AS ?c) WHERE {{ ?a {pred} ?b }}"
    count = list(g.query(q))[0][0]
    print(f"  {label}: {count}")

# Export

print("\n→ Exporting...")

g.serialize("book_kg_enriched.ttl", format="turtle")
print("  ✓ book_kg_enriched.ttl")

inferred_preds = [
    BKG.similarTo,
    BKG.influenceRelevant,
    BKG.transitiveInfluence,
    BKG.similarAuthor,
]
g_inferred = Graph()
for pred in inferred_preds:
    for s, o in g.subject_objects(pred):
        g_inferred.add((s, pred, o))

g_inferred.serialize("inferred_triples.ttl", format="turtle")
print(f"  ✓ inferred_triples.ttl ({len(g_inferred)} triples)")

# Load into GraphDB

print("\n→ Loading inferred triples into GraphDB...")
with open("inferred_triples.ttl", "rb") as f:
    r = requests.post(
        f"{GRAPHDB_BASE}/repositories/{GRAPHDB_REPO}/statements",
        headers={"Content-Type": "text/turtle"},
        data=f
    )
if r.status_code == 204:
    print("  ✓ Inferred triples added to GraphDB")
else:
    print(f"  ✗ GraphDB error: {r.status_code} {r.text}")

print("\n ✓ Reasoning complete.")
