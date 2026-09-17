import subprocess
import time


INTERVAL = 300


def run(command):
    print(f"\nRunning: {command}")
    subprocess.run(command, shell=True, check=True)


def pipeline():
    print("\n=== PIPELINE START ===")

    # Stages 1 and 2
    run("dvc repro")

    # Stage 3
    run("docker compose -f code/deployment/docker-compose.yml up -d --build")

    print("=== PIPELINE COMPLETE ===")


if __name__ == "__main__":
    while True:
        try:
            pipeline()
        except Exception as e:
            print(f"Pipeline failed: {e}")

        print(f"\nWaiting {INTERVAL} seconds...")
        time.sleep(INTERVAL)