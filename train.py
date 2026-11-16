"""Train and evaluate the SymbioNet base network or SymbioNet on Trento, Houston2013 and MUUFL.

Examples:
  python train.py --dataset trento --smoke                     # 2 epochs on part of the test set: checks the setup
  python train.py --dataset trento --exp step1 --runs 3 --sweep data.patch=9,11 data.pca=0,30
  python train.py --dataset trento houston2013 muufl --exp base --runs 5

Settings come from configs/default.yaml, then configs/datasets.yaml (per dataset), then --set (e.g.
--set training.epochs=300 model.decoder=mamba); --sweep runs every combination of the listed values. Every run is
saved to results/<exp>/<dataset>/<setting>/ (run<k>.json with all scores and curves, the weights of the best and
of the last epoch in ckpt/, and the resolved config.yaml; see symbionet/runs.py). A short summary of all settings is
printed at the end and appended to results/<exp>/summary.txt. The experiments of the paper are run by formal.py.
"""
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')       # deterministic cuBLAS; set before CUDA starts

import argparse  # noqa: E402
import copy  # noqa: E402
import itertools  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from symbionet.config import load_yaml, merge, parse_assignments, set_key, setting_name  # noqa: E402
from symbionet.data import DATASETS  # noqa: E402
from symbionet.runs import Setting, find_data_root, run_one, table, write_config, write_summary  # noqa: E402


# ------------------------------------------------------------------------------------------------------ main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', nargs='+', required=True, choices=list(DATASETS))
    p.add_argument('--exp', default='default', help='experiment name (folder under results/)')
    p.add_argument('--config', default=str(ROOT / 'configs' / 'default.yaml'))
    p.add_argument('--data-root', help='folder with Trento/, Houston13/, MUUFL/ (default: data.root)')
    p.add_argument('--runs', type=int, help='runs per setting (default: runs in the config)')
    p.add_argument('--seed', type=int, help='seed of the first run; run k uses seed + k')
    p.add_argument('--set', nargs='*', metavar='KEY=VALUE', help='override settings, e.g. training.epochs=300')
    p.add_argument('--sweep', nargs='*', metavar='KEY=V1,V2', help='run every combination of these values')
    p.add_argument('--device', help='cuda, cpu, ... (default: cuda if available)')
    p.add_argument('--smoke', action='store_true', help='2 epochs, 1 run, 3000 test pixels: checks the setup')
    a = p.parse_args(argv)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    defaults = load_yaml(a.config)
    per_dataset = load_yaml(ROOT / 'configs' / 'datasets.yaml')
    sweep = parse_assignments(a.sweep, many=True)
    settings = [list(zip([k for k, _ in sweep], combo)) for combo in itertools.product(*[v for _, v in sweep])]
    exp = 'smoke' if a.smoke and a.exp == 'default' else a.exp
    out_root = ROOT / 'results' / exp
    print('SymbioNet | experiment %s | device %s | torch %s' % (exp, device, torch.__version__))

    rows = []
    for dataset in a.dataset:
        base = merge(defaults, per_dataset.get(dataset, {}))
        for key, value in parse_assignments(a.set):
            set_key(base, key, value)
        if a.runs is not None:
            base['runs'] = a.runs
        if a.seed is not None:
            base['seed'] = a.seed
        if a.data_root:
            base['data']['root'] = a.data_root
        if a.smoke:
            base['training']['epochs'], base['runs'], base['eval']['subset'], base['print_every'] = 2, 1, 3000, 1
        data_root = find_data_root(base, ROOT, a.data_root)
        scenes = {}
        for pairs in settings:
            cfg = copy.deepcopy(base)
            for key, value in pairs:
                set_key(cfg, key, value)
            st = Setting(cfg, dataset, ROOT, data_root, device, scenes)
            name = setting_name(pairs)
            out = out_root / dataset / name
            write_config(out, cfg)
            print('\n== %s | %s | %s' % (dataset, name, st.scene.summary()))
            print('   ' + st.describe())
            results = [run_one(cfg, st, r, cfg['seed'] + r, device, out, name) for r in range(cfg['runs'])]
            rows.append((dataset, name, write_summary(out, results, st.scene.classes)))
            del st
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    text = table(rows)
    stamp = time.strftime('%Y-%m-%d %H:%M')
    with open(out_root / 'summary.txt', 'a', encoding='utf-8') as f:
        f.write('%s  python train.py %s\n%s\n\n' % (stamp, ' '.join(sys.argv[1:] if argv is None else argv), text))
    print('\n' + '=' * 30 + ' SymbioNet summary (%s, %s) ' % (exp, stamp) + '=' * 30)
    print(text)
    print('OA / AA / Kappa / F1 in %%, mean (± std) over runs; "best" = epoch with the best test OA, "last" = final '
          'epoch. Details: %s' % out_root)


if __name__ == '__main__':
    main()
