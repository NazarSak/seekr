import tempfile
import unittest
from pathlib import Path

from seekr import persist
from seekr.engine import SearchEngine
from seekr.fuzzy import levenshtein
from seekr.index import InvertedIndex
from seekr.snippets import PLAIN, build_snippet
from seekr.tokenizer import analyze, light_stem
from seekr.query import QueryError, parse

_KEEP: list = []
CORPUS = Path(__file__).resolve().parent.parent / "corpus"


def make_engine(docs: dict[str, str]):
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    for name, text in docs.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(text, bytes):
            p.write_bytes(text)
        else:
            p.write_text(text, encoding="utf-8")
    engine, report = SearchEngine.build(root)
    _KEEP.append(tmp)  # keep the temp dir alive for the test run
    return engine, report


class TokenizerTests(unittest.TestCase):
    def test_lowercase_and_punctuation(self):
        self.assertEqual([t for t, _ in analyze("Hello, HELLO! hello.", stem=False, stop=False)], ["hello"] * 3)

    def test_stop_words_dropped_but_positions_kept(self):
        self.assertEqual(list(analyze("search of engines", stem=False)), [("search", 0), ("engines", 2)])

    def test_apostrophes_and_unicode(self):
        self.assertEqual([t for t, _ in analyze("Don’t panic: Київ", stem=False, stop=False)], ["don't", "panic", "київ"])

    def test_stemmer(self):
        for w, s in [("ranking", "rank"), ("ranked", "rank"), ("ranks", "rank"), ("queries", "query"),
                     ("running", "run"), ("indexes", "index"), ("class", "class"), ("bus", "bus")]:
            self.assertEqual(light_stem(w), s, w)


class IndexTests(unittest.TestCase):
    def test_postings_and_stats(self):
        idx = InvertedIndex(stem=False)
        idx.add("a.txt", "A", "red green red")
        idx.add("b.txt", "B", "green blue")
        self.assertEqual(idx.postings["red"], {0: [0, 2]})
        self.assertEqual(idx.df("green"), 2)
        self.assertEqual(idx.tf("red", 0), 2)
        self.assertEqual(idx.n_docs, 2)

    def test_common_word_in_every_doc_is_not_dropped_without_stoplist(self):
        idx = InvertedIndex(stem=False, stop=False)
        for i in range(3):
            idx.add(f"{i}.txt", "t", "the cat")
        self.assertEqual(idx.df("the"), 3)


class QueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e, _ = make_engine({
            "a.txt": "The inverted index maps words to documents. Index everything.",
            "b.txt": "A search engine uses an index. Search engine ranking matters.",
            "c.txt": "Cats and dogs. Nothing about search here, only pets.",
            "d.txt": "Engine search order differs: engine search.",
        })

    def paths(self, q, **kw):
        return [h.doc.path for h in self.e.search(q, limit=10, **kw).hits]

    def test_single_and_and(self):
        self.assertEqual(set(self.paths("index")), {"a.txt", "b.txt"})
        self.assertEqual(self.paths("index search"), ["b.txt"])

    def test_or_not(self):
        self.assertEqual(set(self.paths("cats OR inverted")), {"a.txt", "c.txt"})
        self.assertEqual(set(self.paths("search NOT index")), {"c.txt", "d.txt"})
        self.assertEqual(set(self.paths("search -cats")), {"b.txt", "d.txt"})

    def test_phrase_uses_positions(self):
        self.assertEqual(set(self.paths('"search engine"')), {"b.txt"})  # d.txt has the words in the other order
        self.assertEqual(set(self.paths('"engine search"')), {"d.txt"})

    def test_grouping(self):
        self.assertEqual(set(self.paths("(cats OR inverted) NOT dogs")), {"a.txt"})

    def test_fuzzy_and_prefix(self):
        self.assertIn("a.txt", self.paths("invertd~"))
        self.assertEqual(set(self.paths("invert*")), {"a.txt"})

    def test_case_insensitive(self):
        self.assertEqual(self.paths("INDEX"), self.paths("index"))

    def test_no_results_and_stop_word_only(self):
        r = self.e.search("zzzz")
        self.assertEqual(r.hits, [])
        self.assertEqual(r.note, "no results")
        r = self.e.search("the of and")
        self.assertIn("stop words", r.note)

    def test_bad_syntax_is_reported_not_raised(self):
        for q in ["(index", "index )", "OR", "index OR"]:
            self.assertIn("query error", self.e.search(q).note, q)
        with self.assertRaises(QueryError):
            parse("")


class RankingTests(unittest.TestCase):
    def test_more_occurrences_rank_higher(self):
        e, _ = make_engine({"one.txt": "apple banana cherry date", "many.txt": "apple apple apple banana cherry date"})
        self.assertEqual([h.doc.path for h in e.search("apple").hits][0], "many.txt")

    def test_rare_term_outweighs_common_term(self):
        e, _ = make_engine({
            "a.txt": "common common common rareword",
            "b.txt": "common common common common",
            "c.txt": "common filler text here",
        })
        res = e.search("common OR rareword", explain=True)
        self.assertEqual(res.hits[0].doc.path, "a.txt")

    def test_idf_never_zero_for_term_in_every_doc(self):
        e, _ = make_engine({"a.txt": "shared alpha", "b.txt": "shared beta"})
        res = e.search("shared")
        self.assertTrue(all(h.score > 0 for h in res.hits))

    def test_bm25_and_explanation_sum(self):
        e, _ = make_engine({"a.txt": "alpha beta beta", "b.txt": "alpha gamma"})
        for r in ("tfidf", "bm25"):
            res = e.search("alpha beta", ranker=r, explain=True)
            h = res.hits[0]
            self.assertAlmostEqual(sum(p.weight for p in h.explanation.parts), h.score)


class RobustnessTests(unittest.TestCase):
    def test_empty_non_utf8_duplicates_and_nesting(self):
        e, rep = make_engine({
            "empty.txt": "",
            "blank.txt": "   \n ",
            "latin.txt": "caf\xe9 au lait".encode("latin-1"),
            "x/same.txt": "duplicate name one",
            "y/same.txt": "duplicate name two",
        })
        self.assertEqual({s[0] for s in rep.skipped}, {"empty.txt", "blank.txt"})
        self.assertEqual(rep.loaded, 3)
        self.assertEqual(len(e.search("duplicate", limit=10).hits), 2)  # same filename, distinct paths
        self.assertEqual(e.search("café").total, 1)

    def test_large_file(self):
        e, _ = make_engine({"big.txt": "needle " + "hay " * 500_000})
        self.assertEqual(e.search("needle").total, 1)


class PersistTests(unittest.TestCase):
    def test_roundtrip(self):
        e, _ = make_engine({"a.txt": "alpha beta alpha gamma", "b.txt": "beta delta"})
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "i.json.gz"
            persist.save(e.index, p, root=str(e.root))
            loaded = SearchEngine.load(p)
            self.assertEqual(loaded.index.postings, e.index.postings)
            self.assertEqual(loaded.index.docs, e.index.docs)
            self.assertEqual(loaded.search("alpha").total, 1)


class SnippetTests(unittest.TestCase):
    def test_highlight_and_word_boundaries(self):
        text = "Lorem ipsum dolor sit amet " * 20 + "the needle is here " + "consectetur adipiscing " * 20
        s = build_snippet(text, {"needle"}, stem=False, width=80, markup=PLAIN)
        self.assertIn("**needle**", s)
        body = s.replace("…", "").replace("**", "").split()
        for w in body:
            self.assertIn(w, {"Lorem", "ipsum", "dolor", "sit", "amet", "the", "needle", "is", "here",
                              "consectetur", "adipiscing"}, f"cut word: {w}")

    def test_no_hit_falls_back_to_start(self):
        self.assertTrue(build_snippet("hello world", {"zzz"}, stem=False).startswith("hello"))


class FuzzyTests(unittest.TestCase):
    def test_distance(self):
        self.assertEqual(levenshtein("kitten", "sitting", cap=5), 3)
        self.assertEqual(levenshtein("abc", "abc"), 0)
        self.assertGreater(levenshtein("abc", "xyzxyz", cap=2), 2)


class RealCorpusTests(unittest.TestCase):
    def test_bundled_corpus(self):
        e, rep = SearchEngine.build(CORPUS)
        self.assertGreaterEqual(rep.loaded, 10)
        top = e.search("crawl budget").hits[0]
        self.assertEqual(top.doc.path, "03-crawl-budget.md")
        self.assertIn("canonical", e.search("canonicl~").hits[0].snippet.lower())


def tearDownModule():
    for t in _KEEP:
        t.cleanup()


if __name__ == "__main__":
    unittest.main()
