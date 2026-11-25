import os
from pathlib import Path
import subprocess
import sys

from robotics.generation.generator import main
from robotics.generation.recording import render_images
from robotics.paths import PROJECT_ROOT


def cli():

    if len(sys.argv) == 3 and sys.argv[1] == "--render-images":

        render_images(Path(sys.argv[2]))

    else:

        output = main()

        # Separate EGL rendering process
        render_env = os.environ.copy()
        render_env["MUJOCO_GL"] = "egl"

        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.generate_dataset",
                "--render-images",
                str(output.resolve())
            ],
            cwd=PROJECT_ROOT,
            env=render_env,
            check=True
        )

        print("\nComplete generated episode saved!")


if __name__ == "__main__":
    cli()
