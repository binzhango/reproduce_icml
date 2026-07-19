# /// script
# requires-python = ">=3.11"
# dependencies = ["trackio>=0.10.0"]
# ///
"""Log the complete reproduction bundle as a Trackio artifact."""

from pathlib import Path

import trackio


bundle = Path("artifact_bundle")
trackio.init(project="icml-32495-component-ablation", name="reproduction-bundle")
trackio.log_artifact(bundle, name="icml-32495-reproduction-bundle", type="dataset", aliases=["submission", "latest"])
trackio.finish()
print(f"Logged bundle: {bundle.resolve()}")
