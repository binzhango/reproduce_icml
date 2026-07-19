#!/usr/bin/env python3
"""Attach the complete paper-21218 reproduction bundle to Trackio."""

from pathlib import Path

import trackio


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "artifact_bundle"


def main() -> None:
    trackio.init(
        project="step-repro-21218-artifacts",
        name="complete-reproduction-bundle",
        config={
            "openreview": "4iupzej9nT",
            "official_commit": "03fdea9ac627bb8e6a3f1f5243a1eb6008605198",
            "scope": "released OLMoE 7B-total proxy at 50% expert sparsity",
        },
    )
    artifact = trackio.log_artifact(
        str(BUNDLE),
        name="reproduction-bundle",
        type="dataset",
        aliases=["challenge", "latest"],
    )
    print(f"ARTIFACT_PROJECT={artifact.project}")
    print(f"ARTIFACT_NAME={artifact.name}")
    print(f"ARTIFACT_VERSION={artifact.version}")
    trackio.finish()


if __name__ == "__main__":
    main()
