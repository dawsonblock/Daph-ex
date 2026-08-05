# VOIDED — measured with a defective lexical tokenizer

This retrieval-only pass ran before the fix to
`hrm_adaptive_memory/retrieval/lexical.py::tokenize`. The original pattern
`[A-Za-z0-9_./-]+` treated '.' and '-' as ordinary token characters, so a
sentence-final entity produced the token "plan-000-965." which never matched
the same entity written mid-sentence ("plan-000-965"). Every evidence record
whose entity ended a sentence was therefore invisible to lexical queries,
which depressed the bm25, hash, and all three hybrid arms.

Numbers here understate lexical retrieval and must not be cited. Superseded by
`evidence/gate_b_v2/`. Regression tests:
`tests/unit/test_gate_b_components.py::test_tokenizer_does_not_glue_trailing_punctuation_to_entities`
and `::test_lexical_retrieval_finds_sentence_final_entities`.
