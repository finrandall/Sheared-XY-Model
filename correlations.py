import argparse

import numpy as np


def temporal_rotor_correlation(rotor_angles, max_lag=None):
    samples = rotor_angles.shape[0]

    if max_lag is None or max_lag >= samples:
        max_lag = samples - 1

    fft_length = 1

    while fft_length < 2 * samples:
        fft_length *= 2

    rotors = np.exp(1j * rotor_angles)
    transformed = np.fft.fft(rotors, n=fft_length, axis=0)
    correlation = np.fft.ifft(transformed * np.conjugate(transformed), axis=0).real
    normalisation = np.arange(samples, samples - max_lag - 1, -1) * rotor_angles.shape[1]

    return np.sum(correlation[:max_lag + 1], axis=1) / normalisation


def main():
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Calculate a temporal rotor correlation from a saved trajectory.")
    parser.add_argument("result")
    parser.add_argument("--max-lag", type=int)
    parser.add_argument("--output", default="temporal_correlation.csv")
    args = parser.parse_args()

    data = np.load(args.result)
    trajectory_time = data["trajectory_time"]
    rotor_angles = data["rotor_angle_trajectory"]

    if trajectory_time.shape[0] < 2:
        raise ValueError("This result does not contain a recorded trajectory.")

    correlation = temporal_rotor_correlation(rotor_angles, args.max_lag)
    time_step = np.median(np.diff(trajectory_time))
    lag_time = np.arange(correlation.shape[0]) * time_step

    np.savetxt(args.output, np.column_stack((lag_time, correlation)), delimiter=",", header="lag_time,correlation", comments="")

    plt.plot(lag_time, correlation)
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$C(\tau)$")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
