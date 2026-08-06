# Chapter 04 — Keep a tidy notebook of your experiments

> Matches **Chapter 04** in the book. The same training as before, but the notebook keeps itself.
> The runnable scripts for this chapter are in this folder.

**Labels:** 💻 Runs free on your laptop

---

## The big idea (in plain words)

Back in Chapter 02 you wrote your scores into a text file by hand, one line per run. That works,
but it's easy to forget a line, lose track of *which settings* gave *which score*, or fumble
when you've run twenty experiments.

Imagine instead a smart lab notebook that writes itself. Every time you run an experiment, it
quietly records what dials you turned, what scores you got, and even saves the trained model —
all on its own. Later you open a clean web page, line up all your tries side by side, and pick
the winner. That self-writing notebook is **MLflow**, and at the end you put your best try on a
"trophy shelf" (a **model registry**) so it's easy to find and use later.

## New words (look up anything unfamiliar in the [GLOSSARY](../GLOSSARY.md))

- **MLflow** — a tool that automatically keeps a tidy notebook of every experiment you run.
- **Experiment tracking** — a fancy way of saying "keeping score" of your training runs.
- **Hyperparameter** — a dial you set *before* training (like how many questions the tree may ask).
- **Model registry** — a "trophy shelf" where you store your best model under a name and version.
- **MLflow UI** — a web page that shows all your tracked runs so you can compare them.

## What you will build

You'll train the same fraud model three times with different dial settings, let MLflow record
every run, compare them in a web page, then crown the best one. Each run prints a single line:

```
Recall: 0.800
```

And after crowning the winner:

```
Registered version: 1
```

(Your exact numbers may differ a little — that's normal.)

---

## Let's do it, one small step at a time

### Step 1 — Go to this chapter's project folder

```bash
cd chapter-04-mlflow
```

**What you should see:** your prompt shows you're inside `chapter-04-mlflow`. Type `ls` and you
should see a `src` folder and a `README.md`. The three scripts live inside `src` — run `ls src`
and you'll see `generate_data.py`, `train_with_tracking.py`, and `register_best_model.py`.

> You will run every command from *this* folder (`chapter-04-mlflow`), pointing at the scripts
> with `src/` in front — like `python src/generate_data.py`. Staying in one folder keeps your
> data and your saved experiments together in the same place.

### Step 2 — Make a clean toolbox just for this project

```bash
python3 -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
```

**What you should see:** your prompt now starts with `(venv)`.

### Step 3 — Install the helpers (now including MLflow)

```bash
pip install mlflow scikit-learn pandas numpy
```

**What you should see:** lots of lines, ending in `Successfully installed mlflow... scikit-learn... pandas... numpy...`.
The new one this time is **MLflow** — our self-writing notebook.

### Step 4 — Make your practice data

This chapter comes with its own tiny data maker, so you don't need any other chapter to run it.
Make the table now:

```bash
python src/generate_data.py
```

**What you should see:** `Generated 1000 rows, 18 fraud`. A new file `data/transactions.csv`
now sits right here in this chapter, and the training script will read it.

**What `generate_data.py` does, in plain words** (open it in VS Code to follow along):

1. It invents 1,000 pretend purchases, each with an `amount` and a `time` of day.
2. It marks a purchase as fraud (`is_fraud = 1`) when it's both big and in the dead of night —
   over 500 dollars *and* before 5 a.m. That comes out to 18 fakes out of 1,000.
3. It saves the whole table to `data/transactions.csv` for the next step to read.

> The same random seed is baked in, so you'll get the exact same 1,000 rows every time — handy
> when you want your results to match the book.

### Step 5 — Run three experiments with different dials

Each command trains the model once with different **hyperparameters** (the dials). MLflow
records each run automatically. Run them one at a time:

```bash
python src/train_with_tracking.py --max_depth 3 --min_samples 2
python src/train_with_tracking.py --max_depth 8 --min_samples 5
python src/train_with_tracking.py --max_depth 6 --criterion entropy
```

**What you should see:** each command prints one line like `Recall: 0.800`. You won't see the
scores written to a file — MLflow tucked them away for you behind the scenes.

**What `train_with_tracking.py` does, in plain words** (open it in VS Code to follow along):

1. It opens a list of dials you can set before training: `--max_depth`, `--min_samples`, and
   `--criterion`. These are the **hyperparameters**. If you don't set one, it uses a safe default.
2. It reads the big table of purchases (`transactions.csv`) from this chapter's own `data` folder.
3. It picks two clues — the `amount` and the `time` — and the answer to learn, `is_fraud`.
4. It does the **train/test split**: a big pile to study from, a small pile to quiz with.
5. `mlflow.start_run()` — It tells MLflow, "Start a new try and keep score," then writes down
   which dial settings it used so they're never forgotten.
6. It builds a **decision tree** with your dial settings and lets it study the big pile.
7. It quizzes the model and records three scores — accuracy, recall, and f1 — into MLflow.
8. `mlflow.sklearn.log_model(...)` — It even saves the trained model itself into MLflow, then
   prints the recall score on the screen.

> **The key difference from Chapter 02:** there you wrote one line to `experiments.txt` by hand.
> Here MLflow saves the settings, the three scores, *and* the trained model — automatically,
> every run. No more hand-written log.

#### Why the file lives in `src/`, and what "with_tracking" means

Putting code in a `src/` folder is a common convention — it keeps the source code separate from
data, notebooks, and config so the project stays tidy as it grows. The `with_tracking` half of
the name is the important part: this script is wired up to an **experiment-tracking tool**
(MLflow here; Weights & Biases is another popular choice). Every run logs its dial settings and
its resulting scores somewhere you can line them up side by side later — instead of printing to
the terminal and losing the numbers the moment you close the window.

#### The three dials, one at a time

These flags come from `argparse` and feed straight into scikit-learn's `DecisionTreeClassifier`:

| Flag | What it controls |
| --- | --- |
| `--max_depth` | How many levels of splits the tree may make. Lower = simpler model (risk of **underfitting**); higher = more complex (risk of **overfitting**). Default: `5`. |
| `--min_samples` | Passed to scikit-learn's `min_samples_split` — the fewest rows a node must hold before it's allowed to split. Higher values force simpler, more generalized trees. Default: `2`. |
| `--criterion` | The scoring function for split quality: `gini` (the default) or `entropy` (information gain). Different criteria can grow differently shaped trees on the same data. This one uses `nargs='?'`, so writing `--criterion` with no value means `entropy`. |

#### What the three runs are actually comparing

- **`--max_depth 3 --min_samples 2`** — a shallow, restricted tree. Your simplest, most
  conservative model, and a good baseline.
- **`--max_depth 8 --min_samples 5`** — a much deeper tree, but requiring more rows per split.
  It tests whether extra complexity *with a guardrail* actually improves the scores.
- **`--max_depth 6 --criterion entropy`** — moderate depth, but swaps the split metric from
  `gini` to `entropy`, to see if information-gain-based splits suit this dataset better.

Run together, this is a **manual grid search**: train three variants, then let the tracked
accuracy, recall, and f1 tell you which combination generalizes best. Nothing is automated yet —
you're the one picking the combinations — but MLflow makes the comparison honest.

### Step 6 — Open the notebook web page and compare

Start MLflow's web page:

```bash
mlflow ui
```

**What you should see:** a line like `Listening at: http://127.0.0.1:5000`. Open that address in
your web browser. You'll see your three runs listed.

- Tick the boxes next to all three runs, then click **Compare**.
- Look at the `recall` column to see which dial settings caught the most fraud.

When you're done looking, go back to the terminal and press **Ctrl+C** to stop the web page.

> The web page is just a friendly view of the notebook MLflow has been keeping. Nothing here is
> online or costs money — it's all running on your own computer at the `127.0.0.1` address.

### Step 7 — Put the winner on the trophy shelf

With the web page stopped, crown the best run:

```bash
python src/register_best_model.py
```

**What you should see:**

```
Registered version: 1
```

**What `register_best_model.py` does, in plain words** (open it to follow along):

1. It opens a helper that can read all your saved tries from MLflow.
2. It finds the group of tries named `fraud-detection-v1` (the experiments from Step 5).
3. It asks MLflow for every try, sorted best-to-worst by **recall**. The first one is the winner.
4. It grabs the winner's special address (its "run id").
5. It places that winning model on the **model registry** — the trophy shelf — under the name
   `FraudDetectionModel`.
6. It prints the version number the winner got (version 1, then 2, and so on each new time).

---

## Try it yourself (mini challenges)

- 🔧 **Add a fourth try.** Run `python src/train_with_tracking.py --max_depth 10 --min_samples 10`,
  then refresh the MLflow web page. A new run should appear.
- 🔧 **Change the deepest dial.** Try `--max_depth 1`. In the web page, does its recall drop
  compared with the deeper trees?
- 🔧 **Crown a new winner.** After adding more tries, run `python src/register_best_model.py` again.
  Watch the version number climb to `2`, `3`, and so on.
- 🔧 **Spot the columns.** In the Compare view, find the `max_depth`, `min_samples`, and
  `recall` columns side by side. This is exactly the picture your hand-written log could never show clearly.

## If something breaks

- **`No such file or directory: ... data/transactions.csv`** → You haven't made the data yet.
  Run Step 4 (`python src/generate_data.py`) first.
- **`No module named mlflow`** → Your toolbox isn't on, or MLflow isn't installed. Confirm your
  prompt shows `(venv)`, then redo Step 3 and Step 4.
- **`mlflow: command not found`** → Same fix: make sure `(venv)` is on and you ran the
  `pip install` from Step 4.
- **The web page won't open / "address already in use"** → Another MLflow page may still be
  running. Find the old terminal and press **Ctrl+C**, or just close that terminal, then try again.
- **`register_best_model.py` errors about no runs** → You haven't trained anything yet. Run the
  experiments in Step 5 first.

### Nuclear option — rebuild your toolbox from scratch

If your `venv` gets into a weird state (wrong Python version, half-installed packages, strange
import errors that survive a reinstall), the cleanest fix is to throw the whole `venv` away and
build a fresh one. This deactivates the broken environment, deletes it, recreates it on **Python
3.12**, upgrades `pip`, and reinstalls the three packages this chapter needs.

**Windows (PowerShell):**

```powershell
deactivate
Remove-Item -Recurse -Force .\venv
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python --version
python -m pip --version
cls
python -m pip install --upgrade pip
python -m pip install mlflow pandas scikit-learn
```

**macOS / Linux (bash or zsh):**

```bash
deactivate
rm -rf venv
python3.12 -m venv venv
source venv/bin/activate
python --version
python -m pip --version
clear
python -m pip install --upgrade pip
python -m pip install mlflow pandas scikit-learn
```

**What each line does, in plain words:**

| Step | Windows | macOS / Linux | What it does |
|------|---------|---------------|--------------|
| Turn off the old toolbox | `deactivate` | `deactivate` | Steps out of the current `venv` (safe to ignore if none is active). |
| Delete the old toolbox | `Remove-Item -Recurse -Force .\venv` | `rm -rf venv` | Removes the whole `venv` folder and everything inside it. |
| Build a fresh toolbox | `py -3.12 -m venv venv` | `python3.12 -m venv venv` | Creates a brand-new empty `venv` using Python 3.12. |
| Turn on the new toolbox | `.\venv\Scripts\Activate.ps1` | `source venv/bin/activate` | Activates it — your prompt now starts with `(venv)`. |
| Check the Python version | `python --version` | `python --version` | Confirms you're on Python 3.12. |
| Check pip is present | `python -m pip --version` | `python -m pip --version` | Confirms the package installer is ready. |
| Clear the screen | `cls` | `clear` | Tidies the terminal (purely cosmetic). |
| Upgrade pip | `python -m pip install --upgrade pip` | `python -m pip install --upgrade pip` | Gets the newest installer so packages install cleanly. |
| Reinstall the packages | `python -m pip install mlflow pandas scikit-learn` | `python -m pip install mlflow pandas scikit-learn` | Puts MLflow and its friends back into the fresh toolbox. |

> **Why Python 3.12?** If `py -3.12` (Windows) or `python3.12` (macOS/Linux) says it can't find
> that version, install Python 3.12 first, or swap in whichever 3.11–3.12 you have. Very new or
> very old Python versions sometimes trip up MLflow's dependencies, so pinning a known-good one
> removes a whole class of install headaches.

## A real use case — the MLflow UI dashboard

Everything above records your runs; the **MLflow UI** is where that record pays off. Picture a
teammate asking: *"Which fraud model should we ship — and can you prove it's the best one?"*
Instead of digging through notes, you open the dashboard and answer in about a minute.

Start it from this chapter's folder (with `(venv)` active):

```bash
mlflow ui
```

Then open `http://127.0.0.1:5000` in your browser. Here's how you'd actually use it:

1. **Line up the contenders.** Click into the `fraud-detection-v1` experiment. Every run from
   Step 5 is a row, with its `max_depth`, `min_samples`, `criterion`, and the `accuracy`,
   `recall`, and `f1` scores in plain columns — the side-by-side view a hand-written log could
   never give you.
2. **Sort by what matters.** For fraud, catching real fraud matters most, so click the `recall`
   column to sort highest-first. The winner floats to the top instantly.
3. **Compare head-to-head.** Tick the boxes on two or three runs and click **Compare**. MLflow
   draws the dials and scores next to each other so you can see *exactly* which setting moved the
   needle — e.g. "going from `max_depth 3` to `8` lifted recall but barely touched accuracy."
4. **Prove it and hand it off.** Click the winning run to see its saved model, parameters, and
   metrics on one page. That page *is* your evidence. It's also the run you crown in Step 7 with
   `register_best_model.py`, so the model on the trophy shelf traces straight back to the numbers
   your teammate just saw.

> This is the everyday loop of a real ML engineer: train a few variants, open the dashboard, sort
> by the metric that matters, compare, and register the winner — with a paper trail anyone can check.
> Press **Ctrl+C** in the terminal to stop the dashboard when you're done.

## 4.9–4.11 Guided Labs: The MLflow UI Across Three Industries

*The three labs below (4.9, 4.10, 4.11) reuse the same MLflow setup from earlier in this
chapter, but each one is a new, self-contained script under `usecases/`. The fintech lab
reuses `data/transactions.csv` from Step 4. The healthcare and entertainment labs need two
extra practice datasets — generate them once, from this chapter's folder, before running
either of those labs:*

```bash
python usecases/generate_usecase_data.py
```

*That writes `data/patient_encounters.csv` and `data/subscriber_activity.csv`.*

## 4.9 Guided Lab: Fraud & Credit Risk on the MLflow UI

*You already know how to train a model and watch MLflow write it down. Now
we're going to do something new: walk into the library MLflow built for us
and actually use it — the way a fraud analyst would, on a real Monday
morning. This lab assumes MLflow is already provisioned and `mlflow ui` is
already running at `127.0.0.1:5000`, exactly as you set up earlier in this
chapter.*

---

### 4.9.1 The Scenario

A bank's fraud team retrains its model constantly, because fraud patterns
shift every week. Their whole job comes down to one question, asked every
single morning: **"Of all the real fraud that happened yesterday, how much
did we actually catch?"** That number has a name — **recall** — and it's the
number this entire lab is built around.

### 4.9.2 Script: `fintech_fraud_credit_lab.py`

*This is a new file, separate from anything you've already written this
chapter. Save it as `chapter-04-mlflow/usecases/fintech_fraud_credit_lab.py`.*

```python
import pandas as pd, mlflow, mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score

df = pd.read_csv('data/transactions.csv')
X = df[['amount', 'time']]
y = df['is_fraud']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment('fraud-detection-v1')
with mlflow.start_run():
    mlflow.log_param('max_depth', 6)
    mlflow.log_param('criterion', 'gini')

    clf = DecisionTreeClassifier(max_depth=6, criterion='gini')
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    mlflow.log_metric('accuracy', accuracy_score(y_te, y_pred))
    mlflow.log_metric('recall', recall_score(y_te, y_pred))
    mlflow.log_metric('f1', f1_score(y_te, y_pred))
    mlflow.sklearn.log_model(clf, 'fraud_model')
```

**Now let's read it in small bites.**

**STEP 1 — Load the transactions**

```python
df = pd.read_csv('data/transactions.csv')
X = df[['amount', 'time']]
y = df['is_fraud']
```

You're reading the same transactions table you used earlier in this chapter.
`X` is the clues — amount and time. `y` is the answer — was it fraud, yes or
no.

**STEP 2 — Open a fresh notebook page**

```python
mlflow.set_experiment('fraud-detection-v1')
with mlflow.start_run():
```

This is the same experiment folder you already built earlier in Chapter 4 —
you're simply adding a new page to it.

**STEP 3 — Write down your knobs and grade the quiz**

```python
mlflow.log_param('max_depth', 6)
mlflow.log_metric('recall', recall_score(y_te, y_pred))
mlflow.sklearn.log_model(clf, 'fraud_model')
```

Nothing new here — you've done this before. **Run this script 4—5 times,
changing `max_depth` to 3, then 6, then 10, then 15.** Each run writes a new
row. This is the part that makes the next section actually interesting —
without multiple runs, there's nothing to compare.

---

### 4.9.3 Now Open the UI (this is the new part)

**STEP 1 — Confirm you're in the right room**

Open your browser to `127.0.0.1:5000`. Look at the top of the sidebar. It
must say **"Model training,"** not "GenAI." If it says GenAI, click it to
switch — you have not broken anything, you just walked into the wrong door.

**STEP 2 — Open the training runs table**

Click **Experiments — fraud-detection-v1 — Training runs.** You should see
one row for every time you ran the script — 4 or 5 rows, if you followed
Step 3 above.

**STEP 3 — Choose what you can see**

Click **Columns**, near the top of the table. Turn on `max_depth`,
`criterion`, `recall`, `accuracy`, `f1`. Turn off anything you don't need —
this table is a workbench, not a museum; only keep what helps you decide.

**STEP 4 — Sort by the number that actually matters**

Click the **recall** column header once or twice until you see **DESC**
(highest first). Your best fraud-catcher is now sitting in row one, with zero
math required on your part.

**STEP 5 — See the pattern, not just the numbers**

Click the toggle near the top of the table that switches from **Table** to
**Chart.** Choose a parallel-coordinates or scatter chart with `max_depth` on
one axis and `recall` on the other. Look for the point where the line bends
downward — that's the run where the tree got too deep, started memorizing
instead of learning, and recall dropped. You just found overfitting with your
eyes, not a formula.

**STEP 6 — Ask the table a direct question**

In the **search bar** above the table, type:

```
metrics.recall > 0.85
```

Press enter. The table now shows only the runs a real compliance officer
would consider "good enough to even discuss." This is the exact question a
fraud team asks itself before promoting anything.

> **A Detour You Might Land On**
> If your search bar shows "No runs found" and you're sure you have runs with
> good recall, double-check you didn't accidentally add a stray space or
> quote mark. The search syntax is picky — copy it exactly as written above.

**STEP 7 — Open the winner and look inside**

Click directly on the row-one run. Click the **Artifacts** tab. You'll see
`fraud_model` sitting there — the actual saved file, plus its `MLmodel`
manifest and environment files. This is what "the model" really is: not a
mystery, a folder you can open and read.

---

### 4.9.4 Register and Promote

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
exp = mlflow.get_experiment_by_name('fraud-detection-v1')
runs = client.search_runs(experiment_ids=[exp.experiment_id],
                           order_by=['metrics.recall DESC'])
best_id = runs[0].info.run_id
uri = f'runs:/{best_id}/fraud_model'
result = mlflow.register_model(uri, 'FraudDetectionModel')
```

```bash
mlflow models set-registered-model-alias FraudDetectionModel Champion 1
```

Back in the UI, click **Models — FraudDetectionModel.** You'll see the new
version with a **Champion** badge next to it. Anyone on the team who asks for
"the Champion fraud model" now gets exactly this one, forever, without ever
touching a file path.

---

### 4.9.5 Checkpoint: Can You...

- [ ] Explain, in one sentence, why recall matters more than accuracy for
      catching fraud?
- [ ] Sort the Training runs table by any metric, not just recall?
- [ ] Find the exact run behind a registered model version, using only the
      UI?

### 4.9.6 What Top FinTech Companies Track

Fraud teams retrain constantly because fraud patterns shift; the "best" model
from last week may not be best this week. A real interview question you
should now be able to answer: *"How would you know a fraud model needs
retraining?"* — Answer: you'd watch recall drift across new runs, exactly the
way you just did in this lab.

## 4.10 Guided Lab: Patient Readmission Risk on the MLflow UI

*Same tool, same UI, a different kind of pressure. This lab assumes MLflow is
already provisioned and `mlflow ui` is already running at `127.0.0.1:5000`.*

---

### 4.10.1 The Scenario

A hospital wants to know which patients are likely to be readmitted within 30
days, so care teams can step in early. The stakes here are different from
fraud: **every prediction has to be explainable, a year from now, to an
auditor or a clinical governance board.** MLflow's job in this lab isn't just
tracking a score — it's building the paper trail a hospital is expected to
keep.

The number this whole lab is built around is **AUC** — how well the model
ranks high-risk patients above low-risk ones — because accuracy alone can
hide the fact that a model is failing the sickest patients.

### 4.10.2 Script: `healthcare_readmission_lab.py`

*A new file — save it as
`chapter-04-mlflow/usecases/healthcare_readmission_lab.py`.*

```python
import pandas as pd, mlflow, mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, recall_score, accuracy_score

df = pd.read_csv('data/patient_encounters.csv')
X = df[['age', 'length_of_stay', 'prior_admissions', 'chronic_conditions']]
y = df['readmitted_30d']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment('readmission-risk-v1')
with mlflow.start_run():
    mlflow.set_tag('data_snapshot', '2026-Q2')
    mlflow.set_tag('reviewed_by', 'pending')

    mlflow.log_param('n_estimators', 200)
    mlflow.log_param('max_depth', 8)

    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)[:, 1]

    mlflow.log_metric('accuracy', accuracy_score(y_te, y_pred))
    mlflow.log_metric('auc', roc_auc_score(y_te, y_proba))
    mlflow.log_metric('recall_high_risk', recall_score(y_te, y_pred))
    mlflow.sklearn.log_model(clf, 'readmission_model')
```

**Now let's read it in small bites.**

**STEP 1 — Load the patient encounters**

```python
df = pd.read_csv('data/patient_encounters.csv')
X = df[['age', 'length_of_stay', 'prior_admissions', 'chronic_conditions']]
y = df['readmitted_30d']
```

Same shape as every dataset this chapter: `X` is the clues, `y` is the answer
you want predicted.

**STEP 2 — Tag the run before you do anything else**

```python
mlflow.set_tag('data_snapshot', '2026-Q2')
mlflow.set_tag('reviewed_by', 'pending')
```

This line is new, and it's the single most important habit in this lab. A
**tag** is a sticky note MLflow attaches to the run, separate from your
params and metrics. `data_snapshot` records exactly which version of patient
data trained this model — because six months from now, "which data trained
this" is a question you will be asked, and "I think it was the March data"
is not an acceptable answer in healthcare.

**STEP 3 — Build the model and grade it three ways**

```python
mlflow.log_metric('auc', roc_auc_score(y_te, y_proba))
mlflow.log_metric('recall_high_risk', recall_score(y_te, y_pred))
```

`roc_auc_score` needs probabilities (`y_proba`), not just yes/no predictions
— it's asking "how well did the model rank risky patients above safe ones,"
not just "did it get the final answer right." **Run this script 3—4 times,**
changing `n_estimators` (try 100, 200, 300) and `max_depth` (try 5, 8, 12).

---

### 4.10.3 Now Open the UI

**STEP 1 — Confirm the room**

`127.0.0.1:5000`, sidebar toggle on **"Model training."**

**STEP 2 — Open the table**

**Experiments — readmission-risk-v1 — Training runs.**

**STEP 3 — Turn on the columns that matter here**

Click **Columns**. This time, turn on `data_snapshot` and `reviewed_by`
alongside `auc` and `recall_high_risk`. **Tag columns matter as much as
metric columns in this lab** — that's the healthcare-specific lesson.

**STEP 4 — Sort by AUC, not accuracy**

Click the **auc** column header to sort **DESC**. The run best at correctly
ranking high-risk patients rises to the top — this is a different sort than
the fraud lab, on purpose.

**STEP 5 — Ask a compliance question**

In the search bar, type:

```
tags.data_snapshot = '2026-Q2'
```

This isolates every run trained on one specific data version — exactly the
question an auditor asks: "show me every model ever built on this data."

> **A Detour You Might Land On**
> Tag searches use exact matching with quotes around the value. If you typed
> `tags.data_snapshot = 2026-Q2` without quotes, the search will fail
> silently or error. Quotes are not optional here.

**STEP 6 — Look at the chart view for a health check**

Switch to **Chart**, plot `n_estimators` against `auc`. If the line flattens
out after 200, that's your sign you've hit diminishing returns — more trees
past that point cost compute without buying accuracy.

**STEP 7 — Open the artifacts**

Click into the top run, open **Artifacts**, and confirm `readmission_model`
is sitting there with its full manifest. This is what you'd hand to a
clinical reviewer alongside the prediction — not just a score, the whole
paper trail behind it.

---

### 4.10.4 Register — But Gate It Behind Human Review

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
exp = mlflow.get_experiment_by_name('readmission-risk-v1')
runs = client.search_runs(experiment_ids=[exp.experiment_id],
                           order_by=['metrics.auc DESC'])
best_id = runs[0].info.run_id
uri = f'runs:/{best_id}/readmission_model'
result = mlflow.register_model(uri, 'ReadmissionRiskModel')
```

```bash
mlflow models set-registered-model-alias ReadmissionRiskModel Staging 1
```

Notice this alias says **Staging**, not Champion. In the **Models** tab,
you'll see this version marked as staged — waiting on a clinician panel to
review sample predictions before anyone promotes it further. That pause,
made visible right in the UI, is the entire point of this lab: nothing in
healthcare goes live on "trust me."

---

### 4.10.5 Checkpoint: Can You...

- [ ] Explain why AUC matters more than accuracy for a readmission model?
- [ ] Find every run trained on a specific data snapshot, using only the
      search bar?
- [ ] Explain what a `Staging` alias means, versus a `Champion` alias?

### 4.10.6 What Top Healthcare AI Teams Track

Tagging every run with its exact data snapshot isn't optional polish — in
regulated healthcare work, it's often close to a requirement. A real
interview question you can now answer: *"How would you make sure a clinician
signs off before a risk model goes live?"* — Answer: exactly the alias-based
staging gate you just built.

## 4.11 Guided Lab: Subscriber Churn on the MLflow UI

*Third industry, same habit. This lab assumes MLflow is already provisioned
and `mlflow ui` is already running at `127.0.0.1:5000`.*

---

### 4.11.1 The Scenario

A streaming platform — think Netflix, Peacock, any subscription service —
wants to know which subscribers are about to cancel, early enough to try to
keep them. Every week the viewing data shifts, so this experiment gets
retrained constantly, same as the fraud model back in section 4.9. The number
this lab is built around is **recall on the "about to cancel" group** —
missing a churn-risk subscriber means losing them with zero chance to step
in.

### 4.11.2 Script: `entertainment_churn_lab.py`

*A new file — save it as
`chapter-04-mlflow/usecases/entertainment_churn_lab.py`.*

```python
import pandas as pd, mlflow, mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, recall_score, accuracy_score

df = pd.read_csv('data/subscriber_activity.csv')
X = df[['avg_watch_hours_week', 'days_since_last_login',
        'titles_completed_30d', 'support_tickets_90d']]
y = df['churned_next_30d']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment('subscriber-churn-v1')
with mlflow.start_run():
    mlflow.log_param('n_estimators', 150)
    mlflow.log_param('max_depth', 7)

    clf = RandomForestClassifier(n_estimators=150, max_depth=7, random_state=42)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)[:, 1]

    mlflow.log_metric('accuracy', accuracy_score(y_te, y_pred))
    mlflow.log_metric('auc', roc_auc_score(y_te, y_proba))
    mlflow.log_metric('recall_churn', recall_score(y_te, y_pred))
    mlflow.sklearn.log_model(clf, 'churn_model')
```

**Now let's read it in small bites.**

**STEP 1 — Load subscriber activity**

```python
df = pd.read_csv('data/subscriber_activity.csv')
X = df[['avg_watch_hours_week', 'days_since_last_login',
        'titles_completed_30d', 'support_tickets_90d']]
y = df['churned_next_30d']
```

The clues this time: how much someone watches, how long since they logged
in, how many shows they finished, how many support tickets they filed. The
answer to predict: will they cancel in the next 30 days.

**STEP 2 — Open the experiment**

```python
mlflow.set_experiment('subscriber-churn-v1')
with mlflow.start_run():
```

A brand-new experiment folder, separate from `fraud-detection-v1` and
`readmission-risk-v1` — each business problem gets its own folder, always.

**STEP 3 — Grade it and save it**

```python
mlflow.log_metric('recall_churn', recall_score(y_te, y_pred))
mlflow.sklearn.log_model(clf, 'churn_model')
```

**Run this script 3—4 times,** varying `n_estimators` (try 50, 150, 300) and
`max_depth` (try 4, 7, 10).

---

### 4.11.3 Now Open the UI

**STEP 1 — Confirm the room**

`127.0.0.1:5000`, sidebar toggle on **"Model training."**

**STEP 2 — Open the table**

**Experiments — subscriber-churn-v1 — Training runs.**

**STEP 3 — Choose your columns**

Click **Columns**, turn on `n_estimators`, `max_depth`, `auc`,
`recall_churn`.

**STEP 4 — Sort by the number that matters here**

Click **recall_churn** to sort **DESC** — same instinct as fraud in 4.9:
missing a real churner costs more than a wasted retention email to someone
who was never leaving.

**STEP 5 — Find the diminishing-returns point**

Switch to **Chart**, plot `n_estimators` against `auc` as a scatter. Look for
where the curve goes flat — that's the point where adding more trees stops
helping. A real platform team uses this exact chart to decide how much
compute a retraining run is actually worth.

**STEP 6 — Compare two runs directly**

Select the checkboxes next to your top two runs in the table, then click
**Compare.** MLflow lays their parameters and metrics side by side — this is
how you'd explain to a teammate, in one screenshot, exactly why one run beat
the other.

> **A Detour You Might Land On**
> If Compare shows blank charts, make sure you selected at least two runs
> with the checkboxes first — Compare needs multiple runs selected before it
> has anything to show.

**STEP 7 — Open the artifacts**

Click your top run, open **Artifacts**, confirm `churn_model` is there.

---

### 4.11.4 Register the Winner

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
exp = mlflow.get_experiment_by_name('subscriber-churn-v1')
runs = client.search_runs(experiment_ids=[exp.experiment_id],
                           order_by=['metrics.recall_churn DESC'])
best_id = runs[0].info.run_id
uri = f'runs:/{best_id}/churn_model'
result = mlflow.register_model(uri, 'ChurnRiskModel')
```

```bash
mlflow models set-registered-model-alias ChurnRiskModel Champion 1
```

---

### 4.11.5 One Step Further: A Second System, Same Habit

Streaming platforms run a second model alongside churn — a **recommender**,
deciding what to show you next. You wouldn't train it the same way, but you'd
log it into MLflow the exact same way, just with different metrics:

```python
mlflow.set_experiment('content-recommender-v1')
with mlflow.start_run():
    mlflow.log_param('embedding_dim', 64)
    mlflow.log_metric('precision_at_10', 0.34)  # of top 10 shown, % watched
    mlflow.log_metric('ndcg_at_10', 0.41)        # were the best ranked highest?
```

Back in the UI, you'd sort **Training runs** by `ndcg_at_10 DESC` — same
click, same table, completely different metric. That's the real lesson of
this whole chapter: **the habit never changes. Only your judgment about what
to track does.**

---

### 4.11.6 Checkpoint: Can You...

- [ ] Explain why recall matters for churn the same way it mattered for
      fraud?
- [ ] Use Compare to put two runs side by side?
- [ ] Name one metric a recommendation system would track that a churn model
      never would?

### 4.11.7 What Top Streaming Companies Track

Recommendation quality is measured with precision@k, recall@k, and NDCG@k —
not accuracy, which doesn't even apply to a ranked list of suggestions. A
real interview question you can now answer: *"Why wouldn't you use accuracy
to evaluate a recommender?"* — Answer: accuracy needs a single right answer;
a recommendation is a ranked list, so you need a metric that rewards putting
the best matches near the top, which is exactly what NDCG measures.

## What you just learned

- **MLflow** is a self-writing lab notebook — it does the **experiment tracking** for you.
- **Hyperparameters** are the dials you set before training; MLflow records which ones you used.
- The **MLflow UI** lets you compare many runs side by side in your browser.
- A **model registry** is a trophy shelf for your best model, with a clear name and version.
- This replaces the hand-written log file from Chapter 02 — same idea, far less effort and far
  fewer mistakes.

## Where to next

➡ [Chapter 05 — Let other programs talk to your model](../chapter-05-api). You'll wrap your fraud
model in a "front desk" so other programs can send it a purchase and get back an answer.
