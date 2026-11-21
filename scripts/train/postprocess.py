import numpy as np

from metrics import clean_segments, hysteresis, segments


def gesture_events(predicted, label, min_length, merge_gap):
    return clean_segments(segments(predicted == label), merge_gap, min_length)


def eating_bouts(bites, max_gap):
    bouts = []

    for bite in sorted(bites):
        if bouts and bite[0] - bouts[-1][-1][1] <= max_gap:
            bouts[-1].append(bite)
        else:
            bouts.append([bite])

    return bouts


def eating_speed(bites, sample_rate, max_gap_seconds=60.0, min_bites=3):
    bouts = [
        bout
        for bout in eating_bouts(bites, round(max_gap_seconds * sample_rate))
        if len(bout) >= min_bites
    ]

    bite_count = sum(len(bout) for bout in bouts)
    minutes = sum(bout[-1][1] - bout[0][0] for bout in bouts) / sample_rate / 60
    intervals = [
        (bout[index + 1][0] - bout[index][0]) / sample_rate
        for bout in bouts
        for index in range(len(bout) - 1)
    ]

    return {
        "bouts": len(bouts),
        "bites_in_bouts": bite_count,
        "bout_minutes": minutes,
        "bites_per_minute": bite_count / minutes if minutes else None,
        "median_seconds_between_bites": float(np.median(intervals)) if intervals else None,
    }


def segment_probabilities(starts, probabilities, window_samples, total, stride):
    count = -(-total // stride)
    per_window = window_samples // stride

    sums = np.zeros(count)
    hits = np.zeros(count)

    for start, probability in zip(starts, probabilities):
        first = start // stride
        sums[first:first + per_window] += probability
        hits[first:first + per_window] += 1

    return np.divide(sums, hits, out=np.zeros(count), where=hits > 0)


def reconstruct_meals(
    timeline,
    stride,
    total,
    start_threshold,
    end_threshold,
    merge_gap,
    min_length,
):
    active = hysteresis(timeline, start_threshold, end_threshold)

    return clean_segments(
        [
            (start * stride, min(end * stride, total))
            for start, end in segments(active)
        ],
        merge_gap,
        min_length,
    )
