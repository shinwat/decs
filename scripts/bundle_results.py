import numpy as np
import os
import argparse

def bundle_results(directory):
    success_rates = None
    admittance_rates = None
    for root, _, files in os.walk(directory):
        for file in files:
            data = np.loadtxt(os.path.join(root,file),delimiter=",")
            data = np.expand_dims(data, axis=0)
            if "success" in file:
                success_rates = data if success_rates is None else np.concatenate((success_rates,data))
            elif "admittance" in file:
                admittance_rates = data if admittance_rates is None else np.concatenate((admittance_rates,data))
    ablation = os.path.basename(os.path.abspath(directory))
    if success_rates is not None:
        np.savetxt(os.path.join(directory,ablation+"_success_rates.csv"),success_rates.T,delimiter=",")
    if admittance_rates is not None:
        np.savetxt(os.path.join(directory,ablation+"_admittance_rates.csv"),admittance_rates.T,delimiter=",")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=None,
        help='The directory containing the csv files.'
    )
    args = parser.parse_args()
    bundle_results(args.directory)