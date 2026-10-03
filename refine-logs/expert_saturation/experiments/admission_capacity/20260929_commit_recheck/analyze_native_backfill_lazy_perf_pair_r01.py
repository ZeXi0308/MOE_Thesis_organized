#!/usr/bin/env python3
"""Audit one untimed lazy-ordinary then native-full reference pair.

This seen-input run qualifies an implementation under the original complete-
request, 97% rate, 105% flow and lower-maximum-gap criterion. It does not
establish a new scheduling method or independent confirmation.
"""

import analyze_native_backfill_only_pair_r01 as original


# The original auditor identifies cells by this tuple before applying its
# unchanged native-relative metrics, action-chain and budget checks.
original.ARMS = (("ordinary", "native_full_ordinary_only"),
                 ("native_full", "native_full_reference"))


if __name__ == "__main__":
    original.main()
