"""Fixed grader entry — calls solution.train_and_evaluate; do not repoint to legacy Ridge."""
import json
import os

from data_io import verify_data_dir
from solution import train_and_evaluate


def main():
  metrics = train_and_evaluate(h5dir=verify_data_dir())
  print(json.dumps(metrics))


if __name__ == "__main__":
  main()
