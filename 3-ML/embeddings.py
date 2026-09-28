import json
import random
import torch
import numpy as np
from pathlib import Path
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF, RDFS
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

BKG  = Namespace("http://bookkg.org/ontology#")
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# Load KG
print("→ Loading book_kg_enriched.ttl...")
g = Graph()
g.parse("book_kg_enriched.ttl", format="turtle")
labels = {str(s): str(o) for s, _, o in g.triples((None, RDFS.label, None))}
def get_label(uri): return labels.get(str(uri), str(uri).split("/")[-1])

# Split triples
TRAIN_PREDICATES = [str(BKG.wrote), str(BKG.hasTopic),
                    str(BKG.hasGenre), str(BKG.influencedBy)]

ha_topic_triples, other_triples = [], []
for s, p, o in g:
    if not isinstance(o, URIRef): continue
    if str(p) == str(BKG.hasTopic):
        ha_topic_triples.append((str(s), str(p), str(o)))
    elif str(p) in TRAIN_PREDICATES:
        other_triples.append((str(s), str(p), str(o)))

random.shuffle(ha_topic_triples)
split = int(len(ha_topic_triples) * 0.1)
holdout_triples  = ha_topic_triples[:split]
training_triples = ha_topic_triples[split:] + other_triples

# TriplesFactory
train_arr = np.array([[t[0], t[1], t[2]] for t in training_triples])
test_arr  = np.array([[t[0], t[1], t[2]] for t in holdout_triples])

tf_train = TriplesFactory.from_labeled_triples(train_arr, create_inverse_triples=True)
tf_test  = TriplesFactory.from_labeled_triples(
    test_arr,
    entity_to_id=tf_train.entity_to_id,
    relation_to_id=tf_train.relation_to_id,
    create_inverse_triples=True,
)

# Train TransE 
print("→ Training TransE")
Path("models").mkdir(exist_ok=True)

results = pipeline(
    training=tf_train,
    testing=tf_test,
    model="TransE",
    model_kwargs=dict(embedding_dim=100, scoring_fct_norm=1),
    optimizer="Adam",
    optimizer_kwargs=dict(lr=0.01),
    training_kwargs=dict(num_epochs=100, batch_size=256),
    loss="MarginRankingLoss",
    loss_kwargs=dict(margin=1.0),
    evaluator="RankBasedEvaluator",
    random_seed=SEED,
    device="cpu",
)
results.save_to_directory("models/transe")

#Metrics
print("\n→ Evaluation results:")
metric_dict = results.metric_results.to_dict()

# Print all available keys for debugging
print("  Available metric keys:", list(metric_dict.keys()))


# Try different key formats across PyKEEN versions
def get_metric(d, *keys):
    for key in keys:
        if key in d:
            v = d[key]
            if isinstance(v, dict):
                # drill into both.realistic or both.avg
                for subkey in ["both", "avg", "realistic"]:
                    if subkey in v:
                        v = v[subkey]
                        if isinstance(v, dict):
                            for subkey2 in ["realistic", "avg"]:
                                if subkey2 in v:
                                    return v[subkey2]
                        return v
            return v
    return "N/A"

mrr    = metric_dict["both"]["realistic"]["inverse_harmonic_mean_rank"]
hits1  = metric_dict["both"]["realistic"]["hits_at_1"]
hits10 = metric_dict["both"]["realistic"]["hits_at_10"]
hits3  = metric_dict["both"]["realistic"]["hits_at_3"]

with open("results.json", "w") as f:
    json.dump({"MRR": str(mrr), "Hits@1": str(hits1), "Hits@10": str(hits10),
               "all_metrics": {k: str(v) for k, v in metric_dict.items()}}, f, indent=2)

# True/False positive examples
print("\n→ Link prediction examples:")
model          = results.model
entity_to_id   = tf_train.entity_to_id
relation_to_id = tf_train.relation_to_id
has_topic_rel  = str(BKG.hasTopic)

if has_topic_rel in relation_to_id:
    rel_id = relation_to_id[has_topic_rel]
    for tp in random.sample(holdout_triples, len(holdout_triples)):
        h_id = entity_to_id.get(tp[0])
        t_id = entity_to_id.get(tp[2])
        if h_id is not None and t_id is not None:
            score = model.score_hrt(torch.tensor([[h_id, rel_id, t_id]])).item()
            print(f"\n  TRUE POSITIVE (held-out):")
            print(f"    Book:  {get_label(tp[0])}")
            print(f"    Theme: {get_label(tp[2])}")
            print(f"    Score: {score:.4f}")
            break

    existing = set((tp[0], tp[2]) for tp in ha_topic_triples)
    all_books  = list(set(tp[0] for tp in ha_topic_triples))
    all_themes = list(set(tp[2] for tp in ha_topic_triples))
    for _ in range(300):
        b, t = random.choice(all_books), random.choice(all_themes)
        if (b, t) not in existing and b in entity_to_id and t in entity_to_id:
            score2 = model.score_hrt(
                torch.tensor([[entity_to_id[b], rel_id, entity_to_id[t]]])
            ).item()
            print(f"\n  FALSE POSITIVE CANDIDATE (not in graph):")
            print(f"    Book:  {get_label(b)}")
            print(f"    Theme: {get_label(t)}")
            print(f"    Score: {score2:.4f}")
            break

# Recommendations
print("\n→ Generating recommendations...")
book_theme_counts = {}
for s, p, o in g.triples((None, BKG.hasTopic, None)):
    book_theme_counts[str(s)] = book_theme_counts.get(str(s), 0) + 1

seed_uri   = max(book_theme_counts, key=book_theme_counts.get)
seed_label = get_label(seed_uri)
print(f"  Seed book: {seed_label}")

all_book_uris = list(set(
    str(s) for s, p, o in g
    if str(p) == str(RDF.type) and str(o) == str(BKG.Book)
))

if seed_uri in entity_to_id:
    seed_id  = entity_to_id[seed_uri]
    # Fix: flatten to 1D vector
    seed_vec = model.entity_representations[0](
        torch.tensor([seed_id])
    ).detach().numpy().flatten()

    scores = []
    for book_uri in all_book_uris:
        if book_uri == seed_uri or book_uri not in entity_to_id:
            continue
        b_id  = entity_to_id[book_uri]
        b_vec = model.entity_representations[0](
            torch.tensor([b_id])
        ).detach().numpy().flatten()  # Fix: flatten to 1D
        sim = float(np.dot(seed_vec, b_vec) /
                    (np.linalg.norm(seed_vec) * np.linalg.norm(b_vec) + 1e-9))
        scores.append((book_uri, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  Top-10 recommendations for '{seed_label}':")
    for i, (uri, sim) in enumerate(scores[:10], 1):
        print(f"    {i:2}. {get_label(uri):<50} (sim={sim:.4f})")

print("\nDone")
