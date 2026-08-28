# Artifact policy

`artifacts/` contains generated experiment bundles.  Raw telemetry, activation
captures, model tensors, complete logs, tarballs, source snapshots, and nested
child bundles are local or externally archived evidence; they are not new Git
source files.

The forward policy is:

1. Git tracks this policy and `index.json` only.  Existing tracked run bundles
   are historical evidence and are not deleted or untracked by this policy.
2. `index.json` is the canonical routing layer.  Scientific authority remains
   in `docs/current/`; a bundle or index entry cannot upgrade its verdict.
3. A canonical entry records an immutable manifest SHA-256 and evidence
   boundary.  Child evidence is referenced by manifest hash instead of copied
   recursively into a new bundle.
4. A clean source commit is preferred over repeated source snapshots.  A dirty
   run must preserve that fact and a patch hash; it is not a clean formal run.
5. External storage URIs must be recorded only after upload and checksum
   verification.  Missing external storage is written as `null`, never guessed.
6. Commands and metadata must not contain credentials.  Portable summaries
   should avoid absolute workstation paths and host identifiers.

Removing already tracked bundles or rewriting Git history is a separate,
explicitly reviewed operation.  This policy does neither.
