# VOIDED — protocol deviation found post-run

The post-run receipt check (receipt_check.json) found one violation of the
frozen leak rule: task numeric_derivation-040's B1 prompt contained the gold
answer token "74", created by HF-subword truncation cutting "Station-058-741"
at the 12-token boundary ("Station-058-74"). The B1 answer-leak prefilter ran
on full chunk content only; truncation happened afterward.

Model behavior was unaffected on that task (B1 output 891, quality 0) and no
other receipt violated any assertion, but this run cannot be promoted. The
constructor now re-checks the truncated (derived) content and skips candidates
whose cut creates the answer (fail-closed; regression test
test_b1_truncation_cannot_create_the_answer_token). Superseded by the re-frozen
protocol manifest evidence/gate_a_protocol_manifest_v2.json and the rerun in
evidence/gate_a_qualification_v2r1/.
