import pandas as pd
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL, XSD

# Namespaces
BKG  = Namespace("http://bookkg.org/ontology#")
BOOK = Namespace("http://bookkg.org/book/")
AUTH = Namespace("http://bookkg.org/author/")
THEM = Namespace("http://bookkg.org/theme/")
GENR = Namespace("http://bookkg.org/genre/")

g = Graph()
g.bind("bkg",  BKG)
g.bind("book", BOOK)
g.bind("auth", AUTH)
g.bind("them", THEM)
g.bind("genr", GENR)

# Helper
def uri_safe(s):
    return s.strip().replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")

# Load CSVs
wrote      = pd.read_csv("./data/triples_wrote.csv")
influenced = pd.read_csv("./data/triples_influenced_by.csv")
genre      = pd.read_csv("./data/triples_has_genre.csv")
topics     = pd.read_csv("./data/triples_has_topic.csv")
books      = pd.read_csv("./data/entities_books.csv")
authors    = pd.read_csv("./data/entities_authors.csv")

# Ontology classes
for cls in [BKG.Book, BKG.Author, BKG.Theme, BKG.Genre]:
    g.add((cls, RDF.type, OWL.Class))

for prop, domain, range_ in [
    (BKG.wrote,        BKG.Author, BKG.Book),
    (BKG.hasTopic,     BKG.Book,   BKG.Theme),
    (BKG.hasGenre,     BKG.Book,   BKG.Genre),
    (BKG.influencedBy, BKG.Author, BKG.Author),
    (BKG.similarTo,    BKG.Book,   BKG.Book),
]:
    g.add((prop, RDF.type, OWL.ObjectProperty))
    g.add((prop, RDFS.domain, domain))
    g.add((prop, RDFS.range,  range_))

# similarTo is symmetric
g.add((BKG.similarTo, RDF.type, OWL.SymmetricProperty))

# Books
for _, row in books.iterrows():
    b = BOOK[row["book_id"]]
    g.add((b, RDF.type,      BKG.Book))
    g.add((b, RDFS.label,    Literal(row["book_label"], lang="en")))
    if pd.notna(row.get("pub_year")):
        g.add((b, BKG.pubYear, Literal(str(row["pub_year"]), datatype=XSD.integer)))


for _, row in authors.iterrows():
    a = AUTH[row["author_id"]]
    g.add((a, RDF.type,   BKG.Author))
    g.add((a, RDFS.label, Literal(row["author_label"], lang="en")))

for _, row in wrote.iterrows():
    g.add((AUTH[row["subject"]], BKG.wrote, BOOK[row["object"]]))

for _, row in influenced.iterrows():
    if row["subject"] and row["object"]:
        g.add((AUTH[row["subject"]], BKG.influencedBy, AUTH[row["object"]]))

for _, row in genre.iterrows():
    genre_uri = GENR[uri_safe(row["object"])]
    g.add((genre_uri,         RDF.type,   BKG.Genre))
    g.add((genre_uri,         RDFS.label, Literal(row["object"], lang="en")))
    g.add((BOOK[row["subject"]], BKG.hasGenre, genre_uri))

for _, row in topics.iterrows():
    theme_uri = THEM[uri_safe(row["object"])]
    g.add((theme_uri,            RDF.type,   BKG.Theme))
    g.add((theme_uri,            RDFS.label, Literal(row["object"], lang="en")))
    g.add((BOOK[row["subject"]], BKG.hasTopic, theme_uri))

g.serialize("book_kg.ttl", format="turtle")
print(f"Exported {len(g)} triples to book_kg.ttl")