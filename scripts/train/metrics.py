import numpy as np


def confusion_matrix(true, pred, num_classes):
    index = true.astype(np.int64) * num_classes + pred.astype(np.int64)
    return np.bincount(index, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def per_class_scores(confusion, names):
    scores = {}

    for index, name in enumerate(names):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum() - tp)
        fn = int(confusion[index, :].sum() - tp)
        scores[name] = prf(tp, fp, fn)

    return scores


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def segments(mask):
    mask = np.asarray(mask, dtype=np.int8)
    edges = np.diff(np.concatenate(([0], mask, [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def clean_segments(found, merge_gap=0, min_length=0):
    merged = []

    for start, end in found:
        if merged and start - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    return [(start, end) for start, end in merged if end - start >= min_length]


def match_segments(predicted, actual):
    actual_starts = np.array([start for start, _ in actual], dtype=np.int64)
    actual_ends = np.array([end for _, end in actual], dtype=np.int64)
    used = np.zeros(len(actual), dtype=bool)
    pairs = []
    false_positives = 0

    for start, end in predicted:
        low = int(np.searchsorted(actual_ends, start, side="right"))
        high = int(np.searchsorted(actual_starts, end, side="left"))

        best = None
        best_overlap = 0

        for index in range(low, high):
            if used[index]:
                continue

            overlap = min(end, actual_ends[index]) - max(start, actual_starts[index])

            if overlap > best_overlap:
                best = index
                best_overlap = overlap

        if best is None:
            false_positives += 1
        else:
            used[best] = True
            pairs.append(((start, end), (int(actual_starts[best]), int(actual_ends[best]))))

    return {
        "tp": len(pairs),
        "fp": false_positives,
        "fn": int((~used).sum()),
        "pairs": pairs,
    }


def overlap_length(first, second):
    total = 0
    i = j = 0

    while i < len(first) and j < len(second):
        start = max(first[i][0], second[j][0])
        end = min(first[i][1], second[j][1])

        if end > start:
            total += end - start

        if first[i][1] < second[j][1]:
            i += 1
        else:
            j += 1

    return total


def hysteresis(probabilities, start_threshold, end_threshold):
    active = np.zeros(len(probabilities), dtype=bool)
    inside = False

    for index, value in enumerate(probabilities):
        if inside:
            inside = value >= end_threshold
        else:
            inside = value >= start_threshold

        active[index] = inside

    return active
