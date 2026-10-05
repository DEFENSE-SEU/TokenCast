# Repair canonical catalog lookup

The repository contains a small `catalog_lookup` package. Its catalog lookup
currently mishandles surrounding whitespace, Unicode caseless matching, and
duplicate canonical keys. Repair the implementation while preserving the
public `Catalog`, `Catalog.lookup`, and `DuplicateItemError` names.

The input domain for this task contains non-empty Python `str` item keys and
queries after whitespace trimming. Values may be arbitrary Python objects.
For every item key and lookup query, define its canonical key as:

```python
text.strip().casefold()
```

Implement all three frozen requirements:

- **[R1] surrounding query whitespace:** `Catalog.lookup(query)` must ignore
  all leading and trailing characters removed by Python `str.strip()`. This
  includes non-ASCII whitespace such as U+2003 EM SPACE, for item keys and
  queries alike.
- **[R2] Unicode caseless matching:** lookup must compare canonical keys with
  Python `str.casefold()` semantics, including matches where `str.lower()` is
  insufficient.
- **[R3] canonical collisions:** construction must raise
  `DuplicateItemError` as soon as two supplied item keys have the same
  canonical key. The exception must expose `canonical_key`, `first_key`, and
  `duplicate_key` attributes containing the canonical form, the first raw key,
  and the later raw key. Once the later colliding pair has been requested, the
  constructor must raise before requesting another pair.

The supplied item order defines which raw key is first. Lookup of a missing
canonical key retains the existing `KeyError` behavior. The solution must use
the Python standard library only.

Run the actor-visible tests with:

```text
python -m unittest discover -s tests -v
```

Submit the repaired Python source. Generated environments, caches, and test
output files are outside the deliverable.
