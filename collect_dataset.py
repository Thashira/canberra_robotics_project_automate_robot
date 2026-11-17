from custom_demonstrator import collect_one_episode

NUM_EPISODES = 10

for i in range(NUM_EPISODES):

    print("\n" + "=" * 50)
    print(f"Starting Episode {i + 1}/{NUM_EPISODES}")
    print("=" * 50)

    collect_one_episode()

    if i < NUM_EPISODES - 1:
        input("\nPress Enter to start the next episode...")
        
print("\n10 episodes collection batch completed!")
