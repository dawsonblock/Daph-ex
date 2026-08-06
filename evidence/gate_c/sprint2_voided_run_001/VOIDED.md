# VOIDED — evidence selection dropped already-retrieved second hops

This four-arm run measured a real improvement (one-pass 0.800 → two-pass
0.982), but 9 of 500 tasks failed for a reason internal to the packer rather
than to retrieval, so the arm definitions were not measuring what they claimed.

`TwoPassRetriever` anchored evidence selection on the question's entities
only. The record that *resolves* a bridge names the bridge entity, not the
question's subject — so when pass one already returned both hops (making the
state SUFFICIENT and correctly suppressing a follow-up), the packer then
discarded the second hop as "unanchored". The model consequently answered with
the adapter identifier instead of the category code.

The Gate C taxonomy compounded the confusion by reporting these as
C1_BRIDGE_NOT_DETECTED, because the receipts did not carry first-pass IDs and
the analysis had to substitute selected IDs for them.

Fixes: `EvidenceState.linked_entities` now exposes entities the evidence links
to the question, `TwoPassRetriever` anchors on those as well, and receipts
carry `first_pass_ids`, `second_pass_ids`, `merged_ids`, and `bridge_entities`.
Regression tests: `test_second_hop_survives_packing_when_pass_one_already_found_it`
and `test_linked_entities_are_exposed_for_anchoring`.

Superseded by `evidence/gate_c/sprint2/`. Retained unmodified.
