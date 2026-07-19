#!/usr/bin/env python3
"""Attach the complete reproduction directory as a versioned Trackio artifact."""

from pathlib import Path

import trackio

trackio.init(project="paper579-reproduction", name="final-bundle")
artifact = trackio.log_artifact(
    Path(__file__).resolve().parent,
    name="repro-bundle",
    type="dataset",
    aliases=["final"],
)
trackio.finish()
print(artifact.qualified_name)
