import json
from collections import OrderedDict
import optuna



def safe_load_json(line):
        """Safely load a JSON line, returning None if an error occurs."""
        try:
            return json.loads(line)
        except json.decoder.JSONDecodeError:
            return None


def export_logs(study, output_path, config_model_name):
    trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
    sorted_trials = sorted(trials, key=lambda trial: trial.value, reverse=False)

    best_models_path = f"{output_path}/{config_model_name}_bestThreeModels.txt"
    existing_best_trials = {}
    try:
        with open(best_models_path, "r") as best_models_file:
            existing_best = [safe_load_json(line) for line in best_models_file if line.strip()]
            existing_best = [eb for eb in existing_best if eb]  # Filter out None values
            existing_best_trials = {eb["trial_number"]: eb for eb in existing_best}
    except FileNotFoundError:
        print(f"{best_models_path} not found. Will create a new one.")

    new_best = False
    for trial in sorted_trials[:3]:
        if trial.number not in existing_best_trials or trial.value < existing_best_trials[trial.number].get("value", float('inf')):
            new_best = True
            break

    if new_best:
        with open(best_models_path, "w") as best_models_file:
            for rank, trial in enumerate(sorted_trials[:3], start=1):
                metrics = trial.user_attrs.get("metrics", {})
                trial_info = OrderedDict([
                    ("rank", rank),
                    ("trial_number", trial.number),
                    ("mae", trial.value),
                    ("parameters", trial.params),
                    ("metrics", metrics)
                ])
                best_models_file.write(json.dumps(trial_info, default=str, indent=4) + '\n')

    trial_log_path = f"{output_path}/{config_model_name}.txt"
    highest_logged_trial_number = -1
    try:
        with open(trial_log_path, "r") as infile:
            for line in infile:
                trial_data = safe_load_json(line.strip())
                if trial_data:
                    highest_logged_trial_number = max(highest_logged_trial_number, trial_data["trial_number"])
    except FileNotFoundError:
        print(f"{trial_log_path} not found. Will create a new one.")

    with open(trial_log_path, "a" if highest_logged_trial_number != -1 else "w") as outfile:
        for trial in trials:
            if trial.number > highest_logged_trial_number:
                trial_info = OrderedDict([
                    ("trial_number", trial.number),
                    ("mae", trial.value),
                    ("parameters", trial.params),
                    ("metrics", trial.user_attrs.get("metrics", {}))
                ])
                outfile.write(json.dumps(trial_info, indent=4) + '\n')