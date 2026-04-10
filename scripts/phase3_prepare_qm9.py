from pathlib import Path

from saans_project.baseline import EDMQM9Runtime, load_edm_qm9_config


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_edm_qm9_config(project_root / "configs" / "edm_qm9_smoke.toml")
    runtime = EDMQM9Runtime(cfg, project_root=project_root, device="cpu")
    patched = runtime.patch_vendored_repo()
    runtime.prepare_data()

    print("Patched files:")
    for item in patched:
        print("-", item)
    print("Prepared QM9 files:")
    for split, path in runtime.prepared_qm9_files.items():
        print(f"- {split}: {path} | exists={path.exists()}")
    print("Data splits:", list(runtime.dataloaders.keys()))
    for split in ["train", "valid", "test"]:
        loader = runtime.dataloaders[split]
        print(split, "batches", len(loader), "dataset_size", len(loader.dataset))


if __name__ == "__main__":
    main()
