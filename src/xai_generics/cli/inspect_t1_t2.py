from __future__ import annotations

import argparse
from pathlib import Path

from xai_generics.config import load_yaml
from xai_generics.data.t1_t2_pairs import T1T2PairDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the local T1/T2 dataset loader using raw roots or a manifest."
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=Path("configs/2d/dataset_t1_t2.yaml"),
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config = load_yaml(args.dataset_config)
    if args.manifest is not None:
        dataset = T1T2PairDataset.from_manifest(args.manifest)
        source_name = str(args.manifest)
    else:
        input_root = Path(dataset_config["data"]["training_root"])
        source = dataset_config["modalities"]["source"]
        target = dataset_config["modalities"]["target"]
        dataset = T1T2PairDataset.from_raw_root(input_root, source=source, target=target)
        source_name = str(input_root)

    print(f"Source: {source_name}")
    print(f"Pairs: {len(dataset)}")
    for index in range(min(args.limit, len(dataset))):
        sample = dataset[index]
        print(
            f"[{index}] patient={sample['patient_id']} slice={sample['slice_index']} "
            f"source_shape={sample['source'].shape} target_shape={sample['target'].shape}"
        )


if __name__ == "__main__":
    main()

