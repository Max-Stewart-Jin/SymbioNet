"""Settings. A run is configured by configs/default.yaml, then configs/datasets.yaml (per dataset), then overrides
of the form section.key=value (the command line of train.py, the versions of configs/formal.yaml)."""
import copy
from pathlib import Path

try:
    import yaml
except ImportError:                     # pragma: no cover
    raise SystemExit('PyYAML is needed for the configuration files: pip install pyyaml')


def load_yaml(path) -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def set_key(cfg: dict, dotted: str, value):
    """cfg['a']['b'] = value for 'a.b'; only keys of the default config are accepted (catches typos)."""
    *path, last = dotted.split('.')
    node = cfg
    for k in path:
        if k not in node or not isinstance(node[k], dict):
            raise SystemExit('unknown setting %r' % dotted)
        node = node[k]
    if last not in node:
        raise SystemExit('unknown setting %r' % dotted)
    node[last] = value


def parse_value(text: str):
    """A command-line value read as YAML (9, 0.5, true, null, mamba); numbers such as 1e-3 that YAML 1.1 keeps as
    strings are converted too."""
    value = yaml.safe_load(text)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def parse_assignments(items, many=False):
    """'a.b=v' -> ('a.b', v); with many=True 'a.b=v1,v2' -> ('a.b', [v1, v2])."""
    out = []
    for item in items or []:
        if '=' not in item:
            raise SystemExit('expected key=value, got %r' % item)
        key, text = item.split('=', 1)
        out.append((key, [parse_value(v) for v in text.split(',')] if many else parse_value(text)))
    return out


def setting_name(pairs) -> str:
    return '_'.join('%s%s' % (k.split('.')[-1], v) for k, v in pairs) or 'default'


def dataset_config(root: Path, dataset: str, config=None) -> dict:
    """The default settings of a dataset: configs/default.yaml (or `config`) with configs/datasets.yaml on top."""
    root = Path(root)
    defaults = load_yaml(config or root / 'configs' / 'default.yaml')
    return merge(defaults, load_yaml(root / 'configs' / 'datasets.yaml').get(dataset, {}))


def apply(cfg: dict, pairs) -> dict:
    """A copy of cfg with the (key, value) overrides applied in order."""
    out = copy.deepcopy(cfg)
    for key, value in pairs:
        set_key(out, key, value)
    return out
