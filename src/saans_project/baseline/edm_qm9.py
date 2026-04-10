from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
import sys
import tomllib

import torch


@dataclass
class EDMSchedulerSettings:
    enabled: bool = False
    num_bins: int = 8
    alpha: float = 0.0
    ema_beta: float = 0.95
    baseline_mix_rho: float = 1.0
    hardness_type: str = "coord_only"
    lambda_coord: float = 1.0
    lambda_feat: float = 1.0


@dataclass
class EDMQM9Config:
    dataset: str = "qm9"
    datadir: str = "external/e3_diffusion_for_molecules/qm9/temp"
    batch_size: int = 4
    num_workers: int = 0
    filter_n_atoms: int | None = None
    remove_h: bool = False
    include_charges: bool = True
    conditioning: list[str] = field(default_factory=list)
    model: str = "egnn_dynamics"
    probabilistic_model: str = "diffusion"
    diffusion_steps: int = 100
    diffusion_noise_schedule: str = "polynomial_2"
    diffusion_noise_precision: float = 1e-5
    diffusion_loss_type: str = "l2"
    normalize_factors: list[float] = field(default_factory=lambda: [1.0, 4.0, 1.0])
    n_layers: int = 2
    inv_sublayers: int = 1
    nf: int = 32
    tanh: bool = True
    attention: bool = True
    norm_constant: float = 1.0
    sin_embedding: bool = False
    normalization_factor: float = 1.0
    aggregation_method: str = "sum"
    condition_time: bool = True
    lr: float = 1e-4
    force_download: bool = False
    exp_name: str = "edm_qm9_smoke"
    no_wandb: bool = True
    online: bool = False
    wandb_usr: str = ""
    save_model: bool = False
    generate_epochs: int = 1
    test_epochs: int = 1
    n_report_steps: int = 1
    visualize_every_batch: int = int(1e8)
    augment_noise: float = 0.0
    data_augmentation: bool = False
    break_train_epoch: bool = False
    ode_regularization: float = 1e-3
    resume: str = ""
    start_epoch: int = 0
    ema_decay: float = 0.0
    dp: bool = False
    clip_grad: bool = False
    trace: str = "hutch"
    brute_force: bool = False
    actnorm: bool = True
    no_cuda: bool = True
    scheduler: EDMSchedulerSettings = field(default_factory=EDMSchedulerSettings)


@dataclass
class EDMInstrumentedBatch:
    timestep_int: list[int]
    timestep_normalized: list[float]
    loss_t: list[float]
    error: list[float]
    nll: list[float]


def load_edm_qm9_config(path: str | Path) -> EDMQM9Config:
    with open(path, "rb") as handle:
        raw = tomllib.load(handle)

    edm = raw.get("edm", {})
    scheduler = raw.get("scheduler", {})
    return EDMQM9Config(
        dataset=str(edm.get("dataset", "qm9")),
        datadir=str(edm.get("datadir", "external/e3_diffusion_for_molecules/qm9/temp")),
        batch_size=int(edm.get("batch_size", 4)),
        num_workers=int(edm.get("num_workers", 0)),
        filter_n_atoms=edm.get("filter_n_atoms", None),
        remove_h=bool(edm.get("remove_h", False)),
        include_charges=bool(edm.get("include_charges", True)),
        conditioning=list(edm.get("conditioning", [])),
        model=str(edm.get("model", "egnn_dynamics")),
        probabilistic_model=str(edm.get("probabilistic_model", "diffusion")),
        diffusion_steps=int(edm.get("diffusion_steps", 100)),
        diffusion_noise_schedule=str(edm.get("diffusion_noise_schedule", "polynomial_2")),
        diffusion_noise_precision=float(edm.get("diffusion_noise_precision", 1e-5)),
        diffusion_loss_type=str(edm.get("diffusion_loss_type", "l2")),
        normalize_factors=list(edm.get("normalize_factors", [1.0, 4.0, 1.0])),
        n_layers=int(edm.get("n_layers", 2)),
        inv_sublayers=int(edm.get("inv_sublayers", 1)),
        nf=int(edm.get("nf", 32)),
        tanh=bool(edm.get("tanh", True)),
        attention=bool(edm.get("attention", True)),
        norm_constant=float(edm.get("norm_constant", 1.0)),
        sin_embedding=bool(edm.get("sin_embedding", False)),
        normalization_factor=float(edm.get("normalization_factor", 1.0)),
        aggregation_method=str(edm.get("aggregation_method", "sum")),
        condition_time=bool(edm.get("condition_time", True)),
        lr=float(edm.get("lr", 1e-4)),
        force_download=bool(edm.get("force_download", False)),
        exp_name=str(edm.get("exp_name", "edm_qm9_smoke")),
        no_wandb=bool(edm.get("no_wandb", True)),
        online=bool(edm.get("online", False)),
        wandb_usr=str(edm.get("wandb_usr", "")),
        save_model=bool(edm.get("save_model", False)),
        generate_epochs=int(edm.get("generate_epochs", 1)),
        test_epochs=int(edm.get("test_epochs", 1)),
        n_report_steps=int(edm.get("n_report_steps", 1)),
        visualize_every_batch=int(edm.get("visualize_every_batch", 1e8)),
        augment_noise=float(edm.get("augment_noise", 0.0)),
        data_augmentation=bool(edm.get("data_augmentation", False)),
        break_train_epoch=bool(edm.get("break_train_epoch", False)),
        ode_regularization=float(edm.get("ode_regularization", 1e-3)),
        resume=str(edm.get("resume", "")),
        start_epoch=int(edm.get("start_epoch", 0)),
        ema_decay=float(edm.get("ema_decay", 0.0)),
        dp=bool(edm.get("dp", False)),
        clip_grad=bool(edm.get("clip_grad", False)),
        trace=str(edm.get("trace", "hutch")),
        brute_force=bool(edm.get("brute_force", False)),
        actnorm=bool(edm.get("actnorm", True)),
        no_cuda=bool(edm.get("no_cuda", True)),
        scheduler=EDMSchedulerSettings(
            enabled=bool(scheduler.get("enabled", False)),
            num_bins=int(scheduler.get("num_bins", 8)),
            alpha=float(scheduler.get("alpha", 0.0)),
            ema_beta=float(scheduler.get("ema_beta", 0.95)),
            baseline_mix_rho=float(scheduler.get("baseline_mix_rho", 1.0)),
            hardness_type=str(scheduler.get("hardness_type", "coord_only")),
            lambda_coord=float(scheduler.get("lambda_coord", 1.0)),
            lambda_feat=float(scheduler.get("lambda_feat", 1.0)),
        ),
    )


class EDMQM9Runtime:
    def __init__(self, config: EDMQM9Config, project_root: str | Path | None = None, device: str = "cpu") -> None:
        self.config = config
        self.project_root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[3]
        self.repo_root = self.project_root / "external" / "e3_diffusion_for_molecules"
        self.device = torch.device(device)
        self.dtype = torch.float32
        self._mods: dict[str, ModuleType] = {}
        self.args = self._build_args()
        self.dataset_info = None
        self.dataloaders = None
        self.charge_scale = None
        self.model = None
        self.nodes_dist = None
        self.prop_dist = None
        self.optim = None

    def _build_args(self) -> SimpleNamespace:
        cfg = self.config
        datadir = Path(cfg.datadir)
        if not datadir.is_absolute():
            datadir = (self.project_root / datadir).resolve()
        return SimpleNamespace(
            dataset=cfg.dataset,
            datadir=str(datadir),
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            filter_n_atoms=cfg.filter_n_atoms,
            remove_h=cfg.remove_h,
            include_charges=cfg.include_charges,
            conditioning=list(cfg.conditioning),
            model=cfg.model,
            probabilistic_model=cfg.probabilistic_model,
            diffusion_steps=cfg.diffusion_steps,
            diffusion_noise_schedule=cfg.diffusion_noise_schedule,
            diffusion_noise_precision=cfg.diffusion_noise_precision,
            diffusion_loss_type=cfg.diffusion_loss_type,
            normalize_factors=list(cfg.normalize_factors),
            n_layers=cfg.n_layers,
            inv_sublayers=cfg.inv_sublayers,
            nf=cfg.nf,
            tanh=cfg.tanh,
            attention=cfg.attention,
            norm_constant=cfg.norm_constant,
            sin_embedding=cfg.sin_embedding,
            normalization_factor=cfg.normalization_factor,
            aggregation_method=cfg.aggregation_method,
            condition_time=cfg.condition_time,
            lr=cfg.lr,
            force_download=cfg.force_download,
            exp_name=cfg.exp_name,
            no_wandb=cfg.no_wandb,
            online=cfg.online,
            wandb_usr=cfg.wandb_usr,
            save_model=cfg.save_model,
            generate_epochs=cfg.generate_epochs,
            test_epochs=cfg.test_epochs,
            n_report_steps=cfg.n_report_steps,
            visualize_every_batch=cfg.visualize_every_batch,
            augment_noise=cfg.augment_noise,
            data_augmentation=cfg.data_augmentation,
            break_train_epoch=cfg.break_train_epoch,
            ode_regularization=cfg.ode_regularization,
            resume=cfg.resume,
            start_epoch=cfg.start_epoch,
            ema_decay=cfg.ema_decay,
            dp=cfg.dp,
            clip_grad=cfg.clip_grad,
            trace=cfg.trace,
            brute_force=cfg.brute_force,
            actnorm=cfg.actnorm,
            no_cuda=cfg.no_cuda,
            context_node_nf=0,
        )

    def _ensure_repo_on_path(self) -> None:
        repo = str(self.repo_root)
        if repo not in sys.path:
            sys.path.insert(0, repo)

    def _import_modules(self) -> None:
        if self._mods:
            return
        self._ensure_repo_on_path()
        from configs.datasets_config import get_dataset_info
        from qm9 import dataset as edm_dataset
        from qm9 import losses as edm_losses
        from qm9.models import get_model, get_optim
        from qm9.utils import compute_mean_mad, prepare_context
        from equivariant_diffusion import utils as diffusion_utils

        self._mods = {
            "get_dataset_info": get_dataset_info,
            "dataset": edm_dataset,
            "losses": edm_losses,
            "get_model": get_model,
            "get_optim": get_optim,
            "compute_mean_mad": compute_mean_mad,
            "prepare_context": prepare_context,
            "diffusion_utils": diffusion_utils,
        }

    def patch_vendored_repo(self) -> list[str]:
        replacements: list[tuple[Path, str, str]] = [
            (
                self.repo_root / "qm9" / "data" / "prepare" / "qm9.py",
                "https://springernature.figshare.com/ndownloader/files/3195389",
                "https://ndownloader.figshare.com/files/3195389",
            ),
            (
                self.repo_root / "qm9" / "data" / "prepare" / "qm9.py",
                "https://springernature.figshare.com/ndownloader/files/3195404",
                "https://ndownloader.figshare.com/files/3195404",
            ),
            (
                self.repo_root / "qm9" / "data" / "prepare" / "qm9.py",
                "https://springernature.figshare.com/ndownloader/files/3195395",
                "https://ndownloader.figshare.com/files/3195395",
            ),
            (
                self.repo_root / "qm9" / "data" / "prepare" / "qm9.py",
                "dtype=np.int",
                "dtype=int",
            ),
            (
                self.repo_root / "qm9" / "data" / "prepare" / "md17.py",
                "dtype=np.bool",
                "dtype=bool",
            ),
        ]

        changed: list[str] = []
        for path, old, new in replacements:
            text = path.read_text()
            if old in text:
                path.write_text(text.replace(old, new))
                changed.append(f"patched {path.relative_to(self.project_root)}")
        return changed

    def prepare_data(self) -> None:
        self._import_modules()
        self.dataset_info = self._mods["get_dataset_info"](self.args.dataset, self.args.remove_h)
        self.dataloaders, self.charge_scale = self._mods["dataset"].retrieve_dataloaders(self.args)

    def build_model(self) -> None:
        if self.dataloaders is None:
            self.prepare_data()
        if len(self.args.conditioning) > 0:
            data_dummy = next(iter(self.dataloaders["train"]))
            property_norms = self._mods["compute_mean_mad"](self.dataloaders, self.args.conditioning, self.args.dataset)
            context_dummy = self._mods["prepare_context"](self.args.conditioning, data_dummy, property_norms)
            self.args.context_node_nf = context_dummy.size(2)
        else:
            self.args.context_node_nf = 0

        self.model, self.nodes_dist, self.prop_dist = self._mods["get_model"](self.args, self.device, self.dataset_info, self.dataloaders["train"])
        self.model = self.model.to(self.device)
        self.optim = self._mods["get_optim"](self.args, self.model)

    def prepare(self) -> "EDMQM9Runtime":
        self.patch_vendored_repo()
        self.prepare_data()
        self.build_model()
        return self

    @property
    def prepared_qm9_dir(self) -> Path:
        return self.repo_root / "qm9" / "temp" / "qm9"

    @property
    def prepared_qm9_files(self) -> dict[str, Path]:
        return {
            "train": self.prepared_qm9_dir / "train.npz",
            "valid": self.prepared_qm9_dir / "valid.npz",
            "test": self.prepared_qm9_dir / "test.npz",
        }

    def first_batch(self, split: str = "train") -> dict[str, Any]:
        if self.dataloaders is None:
            self.prepare_data()
        return next(iter(self.dataloaders[split]))

    def unpack_batch(self, data: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor | None]:
        x = data["positions"].to(self.device, self.dtype)
        node_mask = data["atom_mask"].to(self.device, self.dtype).unsqueeze(2)
        edge_mask = data["edge_mask"].to(self.device, self.dtype)
        one_hot = data["one_hot"].to(self.device, self.dtype)
        charges = (data["charges"] if self.args.include_charges else torch.zeros(0)).to(self.device, self.dtype)
        x = self._mods["diffusion_utils"].remove_mean_with_mask(x, node_mask)
        h = {"categorical": one_hot, "integer": charges}
        context = None
        return x, h, node_mask, edge_mask, context

    @torch.no_grad()
    def compute_batch_nll(self, data: dict[str, Any]) -> float:
        if self.model is None:
            self.build_model()
        x, h, node_mask, edge_mask, context = self.unpack_batch(data)
        nll, _, _ = self._mods["losses"].compute_loss_and_nll(self.args, self.model, self.nodes_dist, x, h, node_mask, edge_mask, context)
        return float(nll.item())

    def train_batch_step(self, data: dict[str, Any]) -> float:
        if self.model is None or self.optim is None:
            self.build_model()
        self.model.train()
        x, h, node_mask, edge_mask, context = self.unpack_batch(data)
        self.optim.zero_grad()
        nll, _, _ = self._mods["losses"].compute_loss_and_nll(self.args, self.model, self.nodes_dist, x, h, node_mask, edge_mask, context)
        nll.backward()
        self.optim.step()
        return float(nll.item())

    @torch.no_grad()
    def instrument_batch(self, data: dict[str, Any], t0_always: bool = False) -> EDMInstrumentedBatch:
        if self.model is None:
            self.build_model()
        x, h, node_mask, edge_mask, context = self.unpack_batch(data)
        x, h, delta_log_px = self.model.normalize(x, h, node_mask)
        loss, loss_dict = self.model.compute_loss(x, h, node_mask, edge_mask, context, t0_always=t0_always)
        neg_log_pxh = loss - delta_log_px
        n_nodes = node_mask.squeeze(2).sum(1).long()
        log_pN = self.nodes_dist.log_prob(n_nodes)
        nll = neg_log_pxh - log_pN
        t_int = loss_dict["t"].detach().cpu().long().tolist()
        t_norm = [float(t) / float(self.model.T) for t in t_int]
        return EDMInstrumentedBatch(
            timestep_int=t_int,
            timestep_normalized=t_norm,
            loss_t=[float(x) for x in loss_dict["loss_t"].detach().cpu().tolist()],
            error=[float(x) for x in loss_dict["error"].detach().cpu().tolist()],
            nll=[float(x) for x in nll.detach().cpu().tolist()],
        )
