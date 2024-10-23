import subprocess
import sys
import optuna
import json

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-1)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
    dropout = trial.suggest_uniform('dropout', 0.0, 0.8)

    # Construct the command to run main.py with the suggested hyperparameters
    cmd = [
        sys.executable,  # Path to the Python interpreter
        'main.py',
        '--dataset', 'cora',
        '--method', 'pmlp_gcn',
        '--protocol', 'semi',
        '--lr', str(lr),
        '--weight_decay', str(weight_decay),
        '--dropout', str(dropout),
        '--num_layers', '2',
        '--hidden_channels', '64',
        '--induc',
        '--device', '0',
        '--conv_tr',
        '--conv_va',
        '--conv_te'
    ]

    try:
        # Run main.py as a subprocess
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Parse the output
        for line in result.stdout.strip().split('\n'):
            try:
                output = json.loads(line)
                mean_performance = output['mean_performance']
                mean_performance = float(mean_performance)
                return mean_performance  # Optuna aims to maximize this value
            except json.JSONDecodeError:
                continue  # Not the JSON line we're looking for
        # If the JSON output is not found, raise an exception
        raise ValueError('JSON output not found in main.py output.')
    except subprocess.CalledProcessError as e:
        # If main.py fails, report the error to Optuna and continue
        print(f"An exception occurred while running main.py: {e.stderr}")
        return None  # Optuna will treat this trial as failed

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    print('Best trial:')
    trial = study.best_trial

    print(f'  Mean Performance: {trial.value}')
    print('  Best hyperparameters:')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')
