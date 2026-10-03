# Repeat-localization alignment correction

Date: 2026-08-23

`repeat-divergence-localization-v1.json` is retained but its aggregate
route-to-output ordering verdict is `INVALID_ALIGNMENT`.

The producer stores `full_routes[1:]`. `full_routes[0]` is the last-prompt
forward that produces generated token 0, so retained route step `s` is the
decode forward that produces generated token `s+1`. Version 1 incorrectly
compared `s <= output_token_index`; the correct comparison is
`s + 1 <= output_token_index`.

No raw N0b bundle or sealed campaign report was changed. The corrected
write-once replay is `repeat-divergence-localization-v2.json`. The next GPU
experiment must retain the prompt-tail route and record the forward-to-output
index explicitly.
