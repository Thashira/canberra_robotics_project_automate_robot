import argparse

from robotics.data.collection import collect_one_episode

NUM_EPISODES = 10


def main():
    parser = argparse.ArgumentParser(description="Collect keyboard demonstrations.")
    parser.add_argument(
        "--standalone", action="store_true",
        help="Run the original standalone sequence of two episodes without a batch prompt.",
    )
    if parser.parse_args().standalone:
        collect_one_episode()
        collect_one_episode()
        return

    for i in range(NUM_EPISODES):

        print("\n" + "=" * 50)
        print(f"Starting Episode {i + 1}/{NUM_EPISODES}")
        print("=" * 50)

        collect_one_episode()

        if i < NUM_EPISODES - 1:
            input("\nPress Enter to start the next episode...")

    print("\n10 episodes collection batch completed!")


if __name__ == "__main__":
    main()
