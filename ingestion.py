"""
Book KG — Data Ingestion Pipeline
----------------------------------
Step 1: Pull books, authors, influence chains and OpenLibrary IDs from Wikidata
Step 2: Enrich each book with themes/subjects from OpenLibrary
Step 3: Normalize themes (merge near-duplicates, drop overly specific ones)
Step 4: Export clean triple CSVs ready to load into a KG in data folder

Output files:
  - triples_wrote.csv          (author, wrote, book)
  - triples_influenced_by.csv  (author, influencedBy, author)
  - triples_has_genre.csv      (book, hasGenre, genre)
  - triples_has_topic.csv      (book, hasTopic, theme)
  - entities_books.csv         (book metadata)
  - entities_authors.csv       (author metadata)
"""

import time
import re
import requests
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
import urllib.error

# Configuration

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
OPENLIBRARY_BASE  = "https://openlibrary.org/works"
SPARQL_LIMIT      = 3000       # target number of books
OL_DELAY          = 0.5       # seconds between OpenLibrary calls
MIN_THEME_BOOKS   = 3         # drop themes that appear in fewer than N books
MAX_THEMES_BOOK   = 10        # keep only top N themes per book (after normalization)
OUTPUT_DIR        = "./data"  # directory to write CSVs to

# Step 1

SPARQL_QUERY = """
SELECT DISTINCT
  ?book ?bookLabel
  ?author ?authorLabel
  ?influencedBy ?influencedByLabel
  ?genre ?genreLabel
  ?openLibraryId
  ?pubYear
WHERE {
  VALUES ?type { wd:Q8261 wd:Q7725634 wd:Q47461344 }
  ?book wdt:P31 ?type ;
        wdt:P50  ?author ;
        wdt:P407 wd:Q1860 ;
        wdt:P577 ?pubDate .

  BIND(YEAR(?pubDate) AS ?pubYear)
  FILTER(?pubYear >= 1950)

  OPTIONAL { ?author wdt:P737 ?influencedBy . }   # influenced by
  OPTIONAL { ?book  wdt:P136  ?genre . }           # genre
  OPTIONAL { ?book  wdt:P648  ?openLibraryId . }   # OpenLibrary ID

  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "en" .
  }
}
LIMIT %d
""" % SPARQL_LIMIT

def fetch_wikidata():
    print("→ Querying Wikidata...")
    sparql = SPARQLWrapper(WIKIDATA_ENDPOINT)
    sparql.addCustomHttpHeader("User-Agent", "BookKG-Pipeline/1.0 (e12142151@student.tuwien.ac.at)")
    sparql.setQuery(SPARQL_QUERY)
    sparql.setReturnFormat(JSON)

    for attempt in range(5):
        try:
            results = sparql.query().convert()
            break
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 60 * (attempt + 1)
                print(f"  Rate limited. Waiting {wait}s before retry {attempt+1}/5...")
                time.sleep(wait)
            else:
                raise
    else:
        raise RuntimeError("Wikidata still rate-limiting after 5 attempts.")

    rows = []
    for r in results["results"]["bindings"]:
        rows.append({
            "book_id":             r["book"]["value"].split("/")[-1],
            "book_label":          r.get("bookLabel",         {}).get("value", ""),
            "author_id":           r["author"]["value"].split("/")[-1],
            "author_label":        r.get("authorLabel",       {}).get("value", ""),
            "influenced_by_id":    r.get("influencedBy",      {}).get("value", "").split("/")[-1],
            "influenced_by_label": r.get("influencedByLabel", {}).get("value", ""),
            "genre_id":            r.get("genre",             {}).get("value", "").split("/")[-1],
            "genre_label":         r.get("genreLabel",        {}).get("value", ""),
            "ol_id":               r.get("openLibraryId",     {}).get("value", ""),
            "pub_year":            r.get("pubYear",           {}).get("value", ""),
        })

    df = pd.DataFrame(rows)
    print(f"  ✓ {df['book_id'].nunique()} books, {df['author_id'].nunique()} authors")
    return df


#Step 2

def fetch_ol_themes(ol_id: str) -> list[str]:
    """Return subject list for one OpenLibrary work ID (e.g. OL45804W)."""
    if not ol_id:
        return []
    url = f"{OPENLIBRARY_BASE}/{ol_id}.json"
    try:
        r = requests.get(url, timeout=10,
                         headers={"User-Agent": "BookKG-Pipeline/1.0"})
        if r.status_code != 200:
            return []
        data = r.json()
        return data.get("subjects", [])
    except Exception:
        return []


def enrich_with_themes(df: pd.DataFrame) -> pd.DataFrame:
    """For each unique book with an OL ID, fetch its themes."""
    books_with_ol = df[["book_id", "ol_id"]].drop_duplicates()
    books_with_ol = books_with_ol[books_with_ol["ol_id"] != ""]

    print(f"→ Fetching themes from OpenLibrary for {len(books_with_ol)} books...")
    theme_rows = []
    for i, (_, row) in enumerate(books_with_ol.iterrows()):
        themes = fetch_ol_themes(row["ol_id"])
        for t in themes:
            theme_rows.append({"book_id": row["book_id"], "theme_raw": t})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(books_with_ol)} done...")
        time.sleep(OL_DELAY)

    theme_df = pd.DataFrame(theme_rows)
    print(f"  ✓ {len(theme_df)} raw (book, theme) pairs collected")
    return theme_df


# Step 3

def normalize_theme(t: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    t = t.lower().strip()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


def filter_themes(theme_df: pd.DataFrame) -> pd.DataFrame:
    """
    - Normalize labels
    - Drop themes that appear in fewer than MIN_THEME_BOOKS books
      (too specific to be useful as a connector)
    - Keep at most MAX_THEMES_BOOK per book (ranked by frequency)
    """
    print("→ Normalizing and filtering themes...")
    theme_df = theme_df.copy()
    theme_df["theme"] = theme_df["theme_raw"].apply(normalize_theme)

    # drop rare themes
    counts = theme_df.groupby("theme")["book_id"].nunique()
    common = counts[counts >= MIN_THEME_BOOKS].index
    theme_df = theme_df[theme_df["theme"].isin(common)]

    # rank themes per book by global frequency and keep top N
    theme_df["freq"] = theme_df["theme"].map(counts)
    theme_df = (theme_df
                .sort_values("freq", ascending=False)
                .groupby("book_id")
                .head(MAX_THEMES_BOOK))

    print(f"  ✓ {theme_df['theme'].nunique()} unique themes kept")
    print(f"  ✓ {len(theme_df)} (book, theme) pairs after filtering")
    return theme_df[["book_id", "theme"]]


# Step 4

def export_triples(wikidata_df: pd.DataFrame, theme_df: pd.DataFrame):
    print("→ Exporting triple CSVs...")

    #  wrote 
    wrote = (wikidata_df[["author_id", "book_id"]]
             .drop_duplicates()
             .rename(columns={"author_id": "subject", "book_id": "object"}))
    wrote["predicate"] = "wrote"
    wrote.to_csv(f"{OUTPUT_DIR}/triples_wrote.csv", index=False)
    print(f"  ✓ triples_wrote.csv          ({len(wrote)} triples)")

    #  influencedBy 
    influenced = (wikidata_df[wikidata_df["influenced_by_id"] != ""]
                  [["author_id", "influenced_by_id"]]
                  .drop_duplicates()
                  .rename(columns={"author_id": "subject",
                                   "influenced_by_id": "object"}))
    influenced["predicate"] = "influencedBy"
    influenced.to_csv(f"{OUTPUT_DIR}/triples_influenced_by.csv", index=False)
    print(f"  ✓ triples_influenced_by.csv  ({len(influenced)} triples)")

    #  hasGenre 
    genre = (wikidata_df[wikidata_df["genre_id"] != ""]
             [["book_id", "genre_label"]]
             .drop_duplicates()
             .rename(columns={"book_id": "subject", "genre_label": "object"}))
    genre["predicate"] = "hasGenre"
    genre.to_csv(f"{OUTPUT_DIR}/triples_has_genre.csv", index=False)
    print(f"  ✓ triples_has_genre.csv      ({len(genre)} triples)")

    #  hasTopic 
    topics = (theme_df
              .rename(columns={"book_id": "subject", "theme": "object"}))
    topics["predicate"] = "hasTopic"
    topics.to_csv(f"{OUTPUT_DIR}/triples_has_topic.csv", index=False)
    print(f"  ✓ triples_has_topic.csv      ({len(topics)} triples)")

    # entity tables
    books = (wikidata_df[["book_id", "book_label", "ol_id", "pub_year"]]
             .drop_duplicates("book_id"))
    books.to_csv(f"{OUTPUT_DIR}/entities_books.csv", index=False)
    print(f"  ✓ entities_books.csv         ({len(books)} books)")

    authors = (wikidata_df[["author_id", "author_label"]]
               .drop_duplicates("author_id"))
    authors.to_csv(f"{OUTPUT_DIR}/entities_authors.csv", index=False)
    print(f"  ✓ entities_authors.csv       ({len(authors)} authors)")

    print("\n ✓ Ingestion complete.")


if __name__ == "__main__":
    wikidata_df = fetch_wikidata()
    theme_df    = enrich_with_themes(wikidata_df)
    theme_df    = filter_themes(theme_df)
    export_triples(wikidata_df, theme_df)
