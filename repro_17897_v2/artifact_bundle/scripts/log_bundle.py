#!/usr/bin/env python3
"""Publish the compact v2 reproduction bundle as a Trackio artifact."""

from pathlib import Path

import trackio


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "artifact_bundle"


def main() -> None:
    trackio.init(
        project="ml-agent-repro-17897-v2-artifacts",
        name="verified-reproduction-bundle",
        config={
            "paper": "2505.23723",
            "openreview": "kcPPWaoegr",
            "official_commit": "15932e7525deb99d59f7416bbe8c75077cff3690",
            "t4_job": "6a5a89c6bee6ee1cf4ecddc5",
            "a10g_job": "6a5a8a51bee6ee1cf4ecddcb",
            "scope": "partial reproduction with independent GPU mechanism proxies",
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
