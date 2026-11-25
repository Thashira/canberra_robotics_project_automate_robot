import argparse

from robotics.data.processing import process_episode


def main():
    parser = argparse.ArgumentParser(description="Convert raw demonstration chunks.")
    parser.add_argument("raw_folder")
    parser.add_argument("output_folder")
    args = parser.parse_args()
    process_episode(args.raw_folder, args.output_folder)


if __name__ == "__main__":
    main()
