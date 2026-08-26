# Robot Max recovery

This repository contains the robot source tree, the current `robot max`
launcher and the host helper scripts under `deploy/host-bin/`.

## Rebuild on a new PC

1. Install Ubuntu/ROS 2 Jazzy and the dependencies listed in `package.xml`.
2. Clone this repository and check out the backup branch:

   ```bash
   git clone https://github.com/tattoostamatis-alt/Robot-.git
   cd Robot-/src/home_robot
   git checkout fix/arm-pick-chain
   ```

3. Run `deploy/install.sh`, then build the workspace:

   ```bash
   cd ../..
   colcon build --symlink-install
   source install/setup.bash
   ```

4. Copy `deploy/host-bin/robot` to `/home/<user>/bin/robot`, make it
   executable, and adjust its workspace paths if the username differs.
5. Install the files in `deploy/systemd/` and `deploy/config/` as described
   in `deploy/README.md`; reload systemd and udev.
6. Restore maps separately into `maps/` if they are not already present.
7. Start with `robot max`. The dashboard token is generated locally on first
   start and is intentionally not stored in Git.

## Intentionally not stored here

Secrets and machine-local state are excluded: dashboard tokens, API keys,
passwords, ROS logs, and user home-directory databases. Model weights and
large external dependencies may also need to be downloaded again.
