# Reference values from the formal experiment records

All values below use **five seeds per dataset**. Clean OA is the average over the five seed-level scores.
Damaged OA is the average over the 46 corrupted conditions within each seed, followed by the mean over
the five seeds. The best clean-test checkpoint was selected following the historical paper protocol.
The same model at the final epoch is recorded separately. Units: OA (%) and gain (percentage points).

| Dataset | Base clean | Base damaged | Copies clean | Copies damaged | Full clean | Full damaged |
|---|---:|---:|---:|---:|---:|---:|
| houston2013 | 93.94 | 71.79 | 94.32 | 84.68 | 94.25 | 87.10 |
| muufl | 90.46 | 80.35 | 91.28 | 86.24 | 91.54 | 86.47 |
| trento | 99.31 | 86.30 | 99.38 | 95.28 | 99.33 | 96.99 |

**CNN$_H$ masked-copy transfer (46-condition damaged OA gain, in percentage points):**

| Dataset | CNN$_H$ gain | Our classification backbone gain | Difference |
|---|---:|---:|---:|
| houston2013 | +9.40 | +12.88 | +3.49 |
| muufl | +3.42 | +5.89 | +2.47 |
| trento | +4.06 | +8.98 | +4.92 |

The `Copies` and `Base` rows differ by **masked labeled copies**. The complete `Full` configuration
also includes the gate and the unlabeled masked mutual completion branch. The transfer comparison
does not attribute the gain difference to any individual architectural component.

The external implementations are loaded from their source repositories; their reproduced results are
not included in this repository by default. See [REPRODUCIBILITY](REPRODUCIBILITY.md).

## Archived source records

- [`records/formal/`](../records/formal/) stores the formal run JSON, config YAML, and test-score JSON records.
- [`records/cnn_copies/`](../records/cnn_copies/) stores the CNN$_H$ transfer comparisons.
- The records are **historical observations**, not an independently re-run benchmark in this repository.
- The model was selected by highest *clean test-set OA* every five epochs under the comparison protocol;
  last-epoch results are also recorded. The selector is not an independent validation selector.
