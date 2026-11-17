# Third-party notices

SPADE includes or adapts the following third-party code. Their original license
terms and copyright notices continue to apply.

The full Apache License 2.0 text referenced below is at
[`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).

## RLVE

`spade/external/rlve/` is a vendored subset of
[Zhiyuan-Zeng/RLVE](https://github.com/Zhiyuan-Zeng/RLVE), distributed under the
MIT License. The retained license is at `spade/external/rlve/LICENSE`.

## Berkeley Function Calling Leaderboard

`spade/core/eval/bfcl_ast_checker.py` is a modified, Python-focused adaptation
of the BFCL checker in
[ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla). The upstream
project is distributed under the Apache License 2.0, whose full text is
retained at [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).

## PRIME CodeContests scorer

`spade/core/eval/codecontest_prime_scorer/` contains scorer code copied or
adapted from [PRIME-RL/PRIME](https://github.com/PRIME-RL/PRIME), distributed
under the Apache License 2.0, whose full text is retained at
[`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt). The vendored `pyext`
module also retains its embedded MIT notice.

The exact upstream revisions used for the BFCL and PRIME snapshots were not
recorded when they were copied.

## External datasets

The paper's 7,872-game GPT-5.5 pool is released under Apache-2.0 at
[`spare-rl/spare-gpt55-static-corpus`](https://huggingface.co/datasets/spare-rl/spare-gpt55-static-corpus),
revision `383e4512938c148a7bef2e72b985075eea180647`. Other paper grounding
corpora are not bundled. Their authoritative public snapshots, checksums,
licenses, and redistribution terms remain unresolved and must not be inferred
from the SPADE source license.
