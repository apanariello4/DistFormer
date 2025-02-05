import pickle
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.motsynth import MOTSynth
from main import MODELS
from utils.utils_scripts import is_list_empty


def main(args):
    time_cons_data = {}

    model = MODELS[args.model](args)
    model = model.to(args.device)
    if args.checkpoint:
        model = load_ck(args.checkpoint, model)
    model.eval()

    dataset = MOTSynth(
        args, mode=args.infer_split, return_only_clean=False, no_transforms=True
    )

    test_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        worker_init_fn=dataset.wif_test,
        num_workers=args.n_workers,
        drop_last=False,
        shuffle=False,
    )

    with tqdm(total=len(test_loader), desc="Time Consistency analysis") as pbar:
        for step, sample in enumerate(test_loader):
            (
                x,
                _,
                _,
                video_bboxes,
                distances,
                visibilities,
                _,
                head_coords,
                frame_idx,
                video_idx,
                tracking_ids,
                clip_clean,
            ) = sample

            x = x.to(args.device)
            last_frame_bboxes = [[v[-1] for v in video_bboxes][-1]]

            if is_list_empty(last_frame_bboxes):
                pbar.update()
                continue

            output = model(
                x, [video_bboxes] if args.model == "distformer" else last_frame_bboxes
            )

            gt_distances = distances[0][0].detach().cpu().numpy()
            tracking_ids = tracking_ids[0][0].detach().cpu().numpy()
            pred_distances = output[0] if args.model == "distformer" else output
            pred_distances = pred_distances.detach().cpu().numpy()
            visibilities = visibilities[0][0].detach().cpu().numpy()

            for gt_dist, pred_dist, tracking_id in zip(
                gt_distances, pred_distances, tracking_ids
            ):
                if tracking_id not in time_cons_data:
                    time_cons_data[tracking_id] = {}
                    time_cons_data[tracking_id]["gt_dists"] = []
                    time_cons_data[tracking_id]["pred_dists"] = []
                    time_cons_data[tracking_id]["visibilities"] = []

                time_cons_data[tracking_id]["gt_dists"].append(gt_dist)
                time_cons_data[tracking_id]["pred_dists"].append(pred_dist)
                time_cons_data[tracking_id]["visibilities"].append(visibilities)

            if frame_idx[0][0] == "1799":
                with open(
                    f"{args.performace_save_path}/time_cons_data_{model.__class__.__name__}_{video_idx[0]}.pkl",
                    "wb",
                ) as f:
                    pickle.dump(time_cons_data, f)
                    print(f"Saved time consistency data for sequence {video_idx[0]} \n")

                time_cons_data = {}

            pbar.update()


def load_ck(checkpoint, model) -> None:
    """
    load training checkpoint
    """
    ck = torch.load(Path(checkpoint), map_location=model.device)
    print(f"[loading checkpoint {checkpoint}]")
    weights = ck["weights"] if "weights" in ck else ck["model"]
    state_dict = {k.replace("module.", ""): v for k, v in weights.items()}
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    return model


if __name__ == "__main__":
    from main import parse_args, preprocess_args

    args = parse_args()
    args = preprocess_args(args)

    main(args)
