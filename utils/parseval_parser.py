from copy import deepcopy

import pandas as pd

import wandb

METRICS = [
    "delta_1",
    "abs_rel_diff",
    "squa_rel_diff",
    "rmse_linear",
    "rmse_log",
    "alp_@1m",
]

MODES = {
    "delta_1": "max",
    "abs_rel_diff": "min",
    "squa_rel_diff": "min",
    "rmse_linear": "min",
    "rmse_log": "min",
    "alp_@1m": "max",
}
# "ale_10-20",
# "ale_20-30",
# "ale_30-100",
# "ale_all",
# "aloe_0.0-0.3",
# "aloe_0.3-0.5",
# "aloe_0.5-0.75",
# "ale_0-10"]

PURPLE = "\033[95m"
CYAN = "\033[96m"
DARKCYAN = "\033[36m"
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"
END = "\033[0m"


class Run:
    def __init__(self, entity, project, run_id):
        self.api = wandb.Api()
        self.run = self.api.run(f"{entity}/{project}/{run_id}")
        self._load_summary()
        self.last_epoch = (
            int(self.run.summary["epoch"]) if "epoch" in self.run.summary else None
        )
        self.dataset = self.run.config["dataset"]
        self.config = deepcopy(self.run.config)
        self.entity = self.run.entity
        self.id = self.run.id
        self.project = self.run.project
        self.state = self.run.state
        self.available_classes = self.get_available_classes()
        self.load_all_metrics()
        print(f"Loaded {self}")
        self.cache = {}

    def get_available_classes(self):
        c = self.run.history()
        return [x.split(".")[0] for x in c if x.endswith(".rmse_linear")]

    def _load_summary(self):
        ee = {a: str(b) for a, b in self.run.summary.items()}
        self.summary = {}
        for a, b in ee.items():
            if "grad" in a:
                continue
            try:
                self.summary[a] = eval(b)
            except BaseException:
                self.summary[a] = b

    def get_metrics_per_class(self, class_name=None):
        if class_name in (None, "", "all"):
            return self.run.history(keys=METRICS, pandas=True)
        else:
            if class_name not in self.available_classes:
                return "Class not available"
            df = self.run.history(
                keys=[f"{class_name}.{metric}" for metric in METRICS], pandas=True
            )
            df.columns = [
                c.split(".", 1)[1] if not c.startswith("_") else c for c in df.columns
            ]
            return df

    def load_all_metrics(self):
        for c in self.available_classes:
            setattr(self, f"{c}_metrics", self.get_metrics_per_class(c))

    def get_best_epoch(self, metric="rmse_linear", mode="min", class_name=None):
        m = (
            self.all_metrics
            if class_name in (None, "", "all")
            else getattr(self, f"{class_name}_metrics")
        )
        if mode == "min":
            return m[metric].idxmin()
        elif mode == "max":
            return m[metric].idxmax()

    def print_best_epoch(self, metric="rmse_linear", mode="min", class_name=None):
        best_epoch = self.get_best_epoch(metric, mode, class_name)
        print(f"Best epoch for {metric} is {best_epoch}")
        return self.all_metrics.loc[best_epoch]

    def print_best_epochs_summary(self, metric="rmse_linear", mode="min"):
        print(f"Best epochs for {UNDERLINE}{metric}{END}:")
        print(self.get_best_results_df(metric, mode))
        print("\n")
        self.print_config()

    def get_best_results_df(self, metric="rmse_linear", mode=None):
        if f"get_best_results_df_{metric}_{mode}" in self.cache:
            return self.cache[f"get_best_results_df_{metric}_{mode}"]
        if mode is None:
            mode = MODES.get(metric, None)
        assert mode is not None, f"Mode for {metric} is not defined"

        best_classes = [
            self.get_best_epoch(metric, mode, c) for c in self.available_classes
        ]
        df = pd.concat(
            [
                *[
                    getattr(self, f"{c}_metrics").loc[best_classes[i]]
                    for i, c in enumerate(self.available_classes)
                ]
            ],
            axis=1,
        )

        df.columns = [f"{x} ({y})" for x, y in zip(self.available_classes, df.columns)]
        df = df.transpose()
        df = df.loc[:, ~df.columns.str.startswith("_step")]
        self.cache[f"get_best_results_df_{metric}_{mode}"] = df

        return df

    def get_epoch_results(self, epoch):
        df = pd.concat(
            [
                *[
                    getattr(self, f"{c}_metrics").loc[epoch]
                    for c in self.available_classes
                ]
            ],
            axis=1,
        )

        df.columns = [f"{x} ({y})" for x, y in zip(self.available_classes, df.columns)]
        df = df.transpose()
        df = df.loc[:, ~df.columns.str.startswith("_step")]
        return df

    def print_config(self):
        print(f"Config for {self.entity}/{self.project}/{self.id}:")
        to_print = [
            "lr",
            "backbone",
            "regressor",
            "pool_size",
            "loss",
            "optimizer",
            "scheduler",
            "batch_size",
            "accumulation_steps",
            "loss_warmup",
            "loss_warmup_start",
            "mae_masking",
            "mae_alpha",
        ]
        print(pd.DataFrame({k: [self.config[k]] for k in to_print}))

    def get_config(self):
        to_print = [
            "lr",
            "backbone",
            "regressor",
            "pool_size",
            "loss",
            "optimizer",
            "scheduler",
            "batch_size",
            "accumulation_steps",
            "loss_warmup",
            "loss_warmup_start",
            "mae_masking",
            "mae_alpha",
            "mae_min_bbox_h",
            "mae_min_bbox_w",
        ]
        return pd.DataFrame({k: [self.config[k]] for k in to_print})

    def get_metric_history(self, metric, class_name=None):
        if class_name in (None, "", "all"):
            return self.run.history(keys=metric, pandas=True)
        else:
            if class_name not in self.available_classes:
                return "Class not available"
            df = self.run.history(keys=[f"{class_name}.{metric}"], pandas=True)
            df.columns = [
                c.split(".", 1)[1] if not c.startswith("_") else c for c in df.columns
            ]
            return df

    @staticmethod
    def print_header():
        s = f"{'cls (ep)':10s}"
        for m in METRICS:
            s += f"{m}  "
        print("\n" + BOLD + s + END + "\n")

    @staticmethod
    def print_metrics(metrics, first_column=""):
        s = " " * 10 if first_column == "" else f"{first_column:<10}"
        print(len(s))
        for m in METRICS:
            s += f"{metrics[m]:.4f}" + " " * (len(m) - 4)
        print(s)

    def __repr__(self) -> str:
        return f"Run({self.entity}/{self.project}/{self.id}, dataset={self.dataset}, state={self.state} (last_epoch={self.last_epoch}))"
