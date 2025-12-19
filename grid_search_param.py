

log_dir = '/Users/jil4047/Desktop/myrepos/DICE/log/dice_time_series_top10_60_50'

from pathlib import Path
import csv
import re


def main() -> None:
    base_dir = Path(log_dir)
    file_re = re.compile(r"outcome_k(?P<k>\d+)_hn(?P<hn>\d+)\.log$")
    auc_re = re.compile(r"auc=\s*([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)")

    rows = []
    for log_file in base_dir.glob("outcome_k*_hn*.log"):
        name_match = file_re.match(log_file.name)
        if not name_match:
            continue

        auc_val = None
        match_count = 0
        with log_file.open() as fh:
            for line in fh:
                match = auc_re.search(line)
                if match:
                    match_count += 1
                    if match_count == 3:
                        auc_val = float(match.group(1))
                        break

        if auc_val is None:
            continue

        rows.append(
            (
                int(name_match.group("k")),
                int(name_match.group("hn")),
                auc_val,
            )
        )

    rows.sort(key=lambda item: (item[0], item[1]))

    output_path = base_dir / "A_grid_search_results.csv"
    with output_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["K_clusters", "n_hidden_fea", "AUC"])
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {output_path}")
    # The best result if from:
    best_row = max(rows, key=lambda item: item[2])
    print(f"Best result: K_clusters={best_row[0]}, n_hidden_fea={best_row[1]}, AUC={best_row[2]:.4f}")


if __name__ == "__main__":
    main()
