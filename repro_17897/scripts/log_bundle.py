#!/usr/bin/env python3
"""Attach the complete ML-Agent reproduction bundle as a Trackio artifact."""

from pathlib import Path

import trackio


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "artifact_bundle"


def main() -> None:
    trackio.init(
        project="ml-agent-repro-17897-artifacts",
        name="complete-reproduction-bundle",
        config={
            "paper": "2505.23723",
            "openreview": "kcPPWaoegr",
            "official_commit": "15932e7525deb99d59f7416bbe8c75077cff3690",
            "scope": "partial reproduction with source verification and mechanism proxies",
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
