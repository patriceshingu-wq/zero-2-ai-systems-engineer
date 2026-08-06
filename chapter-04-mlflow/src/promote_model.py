# promote_model.py
# Finds the best run, puts it on the trophy shelf, and sticks a nickname on it.
import argparse
import mlflow
from mlflow.tracking import MlflowClient

parser = argparse.ArgumentParser()
parser.add_argument('--experiment')                  # which group of runs to look in
parser.add_argument('--metric')                      # which score decides the winner
parser.add_argument('--artifact')                    # what the saved model is called inside a run
parser.add_argument('--model', required=True)        # the trophy-shelf name
parser.add_argument('--alias', required=True)        # the nickname to stick on it
# Give --version to move a nickname onto a version that is already on the shelf,
# instead of finding a winner and registering a new one.
parser.add_argument('--version', type=int)
args = parser.parse_args()

client = MlflowClient()

# Short path: just move the nickname, register nothing.
if args.version is not None:
    client.set_registered_model_alias(args.model, args.alias, args.version)
    print(f"Nickname '{args.alias}' now points to "
          f'{args.model} version {args.version}')
    raise SystemExit(0)

missing = [n for n in ('experiment', 'metric', 'artifact')
           if getattr(args, n) is None]
if missing:
    raise SystemExit(
        'Missing ' + ', '.join('--' + n for n in missing) +
        '. Give all three to pick a winner, or give --version to move a '
        'nickname onto a version that already exists.')

exp = mlflow.get_experiment_by_name(args.experiment)
if exp is None:
    raise SystemExit(
        f"No experiment named '{args.experiment}' yet. "
        "Train some runs first, then try again.")

runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    order_by=[f'metrics.{args.metric} DESC'])
if not runs:
    raise SystemExit(
        f"'{args.experiment}' has no runs in it yet. "
        "Train some runs first, then try again.")

best = runs[0]
score = best.data.metrics.get(args.metric)
if score is None:
    raise SystemExit(
        f"These runs have no score called '{args.metric}'. "
        f"Scores they do have: {', '.join(sorted(best.data.metrics)) or 'none'}")

uri = f'runs:/{best.info.run_id}/{args.artifact}'
result = mlflow.register_model(uri, args.model)
client.set_registered_model_alias(args.model, args.alias, result.version)

print(f'Best run: {args.metric} = {score:.3f}')
print(f'Registered {args.model} version {result.version}')
print(f"Nickname '{args.alias}' now points to version {result.version}")
