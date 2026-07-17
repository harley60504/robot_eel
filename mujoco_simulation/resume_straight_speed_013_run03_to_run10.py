from __future__ import annotations

from pathlib import Path

import eel_pipeline_gui as gui


class Var:
    def __init__(self, value: str) -> None:
        self.value = value

    def get(self) -> str:
        return self.value

    def set(self, value: str) -> None:
        self.value = value


class BoolVar:
    def __init__(self, value: bool) -> None:
        self.value = value

    def get(self) -> bool:
        return self.value


class Root:
    def after(self, _delay: int, callback=None):
        if callback is not None:
            callback()


class Logger:
    def write(self, message: str) -> None:
        print(message, end="", flush=True)


def make_runner() -> gui.EelPipelineGui:
    runner = gui.EelPipelineGui.__new__(gui.EelPipelineGui)
    runner.root = Root()
    runner.logger = Logger()
    runner.worker_running = False

    runner.out_var = Var(str(gui.DEFAULT_PIPELINE_ROOT))
    runner.rl_train_mode_var = Var("straight")
    runner.rl_target_speed_var = Var("0.13")
    runner.rl_frequency_low_var = Var("0.7")
    runner.rl_frequency_high_var = Var("1.2")
    runner.rl_phase_lag_low_var = Var("0.3")
    runner.rl_phase_lag_high_var = Var("0.8")
    runner.rl_train_timesteps_var = Var("200000")
    runner.rl_eval_freq_var = Var("5000")
    runner.rl_load_model_var = Var("")
    runner.rl_reward_average_seconds_var = Var("")

    runner.rl_turn_direction_var = Var("right")
    runner.rl_target_yaw_rate_var = Var("0.45")
    runner.rl_target_radius_var = Var("0.4")
    runner.rl_use_yaw_reward_var = BoolVar(True)
    runner.rl_use_radius_reward_var = BoolVar(True)
    runner.rl_yaw_reward_weight_var = Var("1.20")
    runner.rl_radius_reward_weight_var = Var("1.20")
    runner.rl_bias_low_var = Var("-0.35")
    runner.rl_bias_high_var = Var("0.35")
    runner.rl_freq_var = Var("")
    runner.rl_wavelength_var = Var("")
    runner.rl_ajoint_var = Var("")
    runner.rl_boundary_x_min_var = Var("")
    runner.rl_boundary_x_max_var = Var("")
    runner.rl_boundary_y_var = Var("")

    runner.gait_json_var = Var("")
    runner.rl_model_var = Var("")
    runner.rl_output_json_var = Var(str(gui.RL_GAIT_DIR / "straight_speed_013.json"))
    runner.rl_train_output_var = Var(str(gui.ZIP_DIR / "straight_speed_013.zip"))
    runner.view_model_var = Var("")
    runner.view_summary_var = Var("")
    runner.view_gait_var = Var("")
    return runner


def main() -> None:
    runner = make_runner()
    print("Resume straight_speed_013")
    print("target_speed=0.13")
    print("freq_low=0.7 freq_high=1.2")
    print("phase_lag_low=0.3 phase_lag_high=0.8")
    print("timesteps=200000 eval_freq=5000")

    for index in range(3, 11):
        run_name = f"straight_speed_013_run{index:02d}"
        model_zip = gui.ZIP_DIR / f"{run_name}.zip"
        organized_manifest = runner.organized_run_dir(run_name) / "organized_outputs_manifest.csv"
        if model_zip.exists() and organized_manifest.exists():
            print(f"\n=== Skip existing complete run: {run_name} ===")
            continue

        output_json = gui.RL_GAIT_DIR / f"{run_name}.json"
        output_base = gui.ZIP_DIR / f"{run_name}.zip"
        cmd = runner._build_train_command(output_base)
        print(f"\n=== RL PPO training {index}/10: {run_name} ===")
        print("CMD: " + " ".join(cmd))
        runner._run_command_stream(cmd)

        model_zip = runner._training_output_zip_for_base(Path(output_base))
        print(f"training output model: {model_zip}")
        if not model_zip.exists():
            raise FileNotFoundError(f"Training finished but model zip was not found: {model_zip}")

        runner._run_free_swim_post_train_outputs(run_name, model_zip, output_json)
        runner.organize_run_outputs(run_name, model_zip, output_json)

    print("\n=== Done: straight_speed_013 run03-run10 ===")


if __name__ == "__main__":
    main()
