# Chapter 04 — Keep a tidy notebook of your experiments

> Matches **Chapter 04** in the book. Same training as before, but now the notebook writes itself.
> The scripts for this chapter are in this folder.

**Labels:** 💻 Runs free on your laptop

---

## The big idea

Back in Chapter 02, you wrote your scores into a text file by hand, one line per run. That
works, but it's easy to forget a line, mix up which settings gave which score, or lose track
once you've run twenty experiments.

Now imagine a notebook that writes itself. Every time you train a model, it quietly writes down
every setting you used, every score you got, and even saves a copy of the trained model. You
don't have to do anything. Later, you open a clean web page, see every attempt lined up in a
table, and pick the winner just by looking. That self-writing notebook is called **MLflow**.

Once you've picked a winner, you put it on a "trophy shelf." MLflow calls this a **model
registry**. It lets anyone on your team find "the best one" later without digging through files.

Later in this chapter, you'll use this same trick on three real jobs a company might pay you to
do: catching bank fraud, spotting hospital patients who might come back too soon, and finding
streaming subscribers who are about to quit. Same tool, three different problems. That's the
whole point of learning it well here.

## New words (look up anything unfamiliar in the [GLOSSARY](../GLOSSARY.md))

- **MLflow** — a tool that automatically keeps a tidy notebook of every experiment you run.
- **Experiment tracking** — a fancy way of saying "keeping score" of your training runs.
- **Hyperparameter** — a setting you choose *before* training (like how many questions the tree may ask).
- **Model registry** — a "trophy shelf" where you store your best model under a name and version.
- **MLflow UI** — a web page that shows all your tracked runs so you can compare them.
- **Tag** — a sticky note you stick on a run to remember something about it later.
- **Alias** — a nickname (like "Champion" or "Staging") you stick on a model version, so people
  always know which one to use without remembering a long file path.
- **Recall** — out of everything bad that really happened, how much did the model actually catch?
- **AUC** — how good the model is at ranking risky things above safe things, not just guessing yes or no.

## What you will build

You'll train the same fraud model three times, each time with different settings. MLflow
records every run. You'll compare them in a web page, then pick the best one. Each run prints
one line:

```
Recall: 0.800
```

And after you crown the winner:

```
Registered version: 1
```

(Your exact numbers may look a little different. That's normal.)

Later in the chapter, you'll do this whole thing two more times, on two brand-new problems: a
hospital dataset and a streaming-service dataset. Repeating it helps the pattern stick.

---

## Let's do it, one small step at a time

### Step 1 — Go to this chapter's project folder

```bash
cd chapter-04-mlflow
```

**What you should see:** your prompt shows you're inside `chapter-04-mlflow`. Type `ls` and you
should see a `src` folder and a `README.md`. The scripts live inside `src`. Run `ls src` and
you'll see `generate_data.py`, `train_with_tracking.py`, `register_best_model.py`, and
`promote_model.py`.

> Run every command from *this* folder (`chapter-04-mlflow`), and point at the scripts with
> `src/` in front — like `python src/generate_data.py`. Staying in one folder keeps your data
> and your saved experiments together in the same place.

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

**What you should see:** `Generated 1000 rows, 18 fraud`. A new file, `data/transactions.csv`,
now sits in this chapter's folder. The training script will read it.

**What `generate_data.py` does, in plain words** (open it in VS Code to follow along):

1. It invents 1,000 pretend purchases, each with an `amount` and a `time` of day.
2. It marks a purchase as fraud (`is_fraud = 1`) when it's both big and late at night — over
   500 dollars *and* before 5 a.m. That comes out to 18 fakes out of 1,000.
3. It saves the whole table to `data/transactions.csv` for the next step to read.

> The same random seed is baked in, so you get the exact same 1,000 rows every time. That's
> handy when you want your results to match the book.

### Step 5 — Run three experiments with different settings

Each command trains the model once, using different **hyperparameters** (settings). MLflow
records each run automatically. Run them one at a time:

```bash
python src/train_with_tracking.py --max_depth 3 --min_samples 2
python src/train_with_tracking.py --max_depth 8 --min_samples 5
python src/train_with_tracking.py --max_depth 6 --criterion entropy
```

**What you should see:** each command prints one line, like `Recall: 0.800`. You won't see the
scores written to a file — MLflow saved them for you behind the scenes.

**What `train_with_tracking.py` does, in plain words** (open it in VS Code to follow along):

1. It offers a list of settings you can choose before training: `--max_depth`, `--min_samples`,
   and `--criterion`. These are the **hyperparameters**. If you skip one, it uses a safe default.
2. It reads the big table of purchases (`transactions.csv`) from this chapter's `data` folder.
3. It picks two clues — the `amount` and the `time` — and the answer to learn, `is_fraud`.
4. It splits the table into two piles: a big pile to learn from, and a small pile to quiz with
   afterward. That way the model is tested on purchases it never saw while learning.
5. `mlflow.start_run()` tells MLflow "start a new page in the notebook." Everything that
   happens next gets written down automatically, including which settings you used.
6. It builds a **decision tree** — think of it as a big game of twenty questions the computer
   plays against each purchase, ending in a "fraud" or "not fraud" guess — using the settings
   you chose, and lets it learn from the big pile.
7. It quizzes the model on the small pile and writes three scores into the notebook: accuracy,
   recall, and f1.
8. `mlflow.sklearn.log_model(...)` saves the whole trained model into the notebook, not just its
   scores, then prints the recall score on the screen.

> **The key difference from Chapter 02:** back then, you wrote one line to `experiments.txt` by
> hand. Here, MLflow saves the settings, the three scores, *and* the trained model —
> automatically, every run. No more hand-written log.

#### Why the file lives in `src/`, and what "with_tracking" means

Putting code in a `src/` folder is a common habit. It keeps the actual program code separate
from data, notes, and settings, so the project stays tidy as it grows. The `with_tracking` part
of the name is the important bit: this script is connected to a notebook-keeping tool (MLflow
here — some teams use a similar tool called Weights & Biases). Every run writes its settings and
scores somewhere you can compare later, instead of printing to the screen and losing the numbers
the moment you close the window.

#### The three settings, one at a time

| Setting | What it does |
| --- | --- |
| `--max_depth` | How many questions deep the tree can go before it must guess. A small number keeps things simple but might miss patterns. A big number can get too clever and start memorizing instead of learning. Default: `5`. |
| `--min_samples` | The smallest group of purchases the tree is allowed to split into two smaller groups. A bigger number keeps the tree simpler and stops it from chasing tiny, one-off patterns. Default: `2`. |
| `--criterion` | The rule the tree uses to pick its next question: `gini` (the default) or `entropy`. Both are different ways of asking "which question best separates the good stuff from the bad stuff?" They usually agree, but not always. Leaving this out defaults to `entropy` in this chapter's examples. |

#### What the three runs are actually comparing

- **`--max_depth 3 --min_samples 2`** — a short, simple tree. Your safest, most cautious guess,
  and a good baseline to compare everything else against.
- **`--max_depth 8 --min_samples 5`** — a much deeper tree, but with a rule that stops it from
  splitting on tiny groups. This checks whether going deeper actually helps, or just adds noise.
- **`--max_depth 6 --criterion entropy`** — a medium-depth tree that picks its questions with the
  other rule (`entropy` instead of `gini`), to see if that changes anything.

Running all three and comparing them is like trying on three outfits before you leave the house.
Nothing fancy is automated yet — you're just trying things on purpose and keeping honest notes on
which one fits best.

### Step 6 — Open the notebook web page and compare

Start MLflow's web page:

```bash
mlflow ui
```

**What you should see:** a line like `Listening at: http://127.0.0.1:5000`. Open that address in
your web browser. You'll see your three runs listed.

- Tick the boxes next to all three runs, then click **Compare**.
- Look at the `recall` column to see which settings caught the most fraud.

When you're done looking, go back to the terminal and press **Ctrl+C** to stop the web page.

> The web page is just a friendly view of the notebook MLflow has been keeping. Nothing here is
> online or costs money — it all runs on your own computer, at the `127.0.0.1` address.

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

1. It opens a helper that can read every saved run out of MLflow's notebook.
2. It finds the group of runs named `fraud-detection-v1` (the ones you made in Step 5).
3. It asks for all of them back, sorted best-to-worst by **recall**. Whichever run is listed
   first is the winner.
4. It grabs that winning run's special ID number, like grabbing a claim ticket.
5. It places that winning model on the trophy shelf under the name `FraudDetectionModel`.
6. It prints the version number the winner just got — version 1 the first time, then 2, 3, and
   so on, each time you crown a new winner.

---

## Try it yourself (mini challenges)

- 🔧 **Add a fourth try.** Run `python src/train_with_tracking.py --max_depth 10 --min_samples 10`,
  then refresh the MLflow web page. A new run should appear.
- 🔧 **Change the deepest setting.** Try `--max_depth 1`. In the web page, does its recall drop
  compared with the deeper trees?
- 🔧 **Crown a new winner.** After adding more tries, run `python src/register_best_model.py` again.
  Watch the version number climb to `2`, `3`, and so on.
- 🔧 **Spot the columns.** In the Compare view, find the `max_depth`, `min_samples`, and
  `recall` columns side by side. This is exactly the picture a hand-written log could never show
  as clearly.

## If something breaks

- **`No such file or directory: ... data/transactions.csv`** → You haven't made the data yet.
  Run Step 4 (`python src/generate_data.py`) first.
- **`No module named mlflow`** → Your toolbox isn't on, or MLflow isn't installed. Confirm your
  prompt shows `(venv)`, then redo Step 3 and Step 4.
- **`mlflow: command not found`** → Same fix: make sure `(venv)` is on, and that you ran the
  `pip install` from Step 3.
- **The web page won't open / "address already in use"** → Another MLflow page may still be
  running. Find the old terminal and press **Ctrl+C**, or close that terminal, then try again.
- **`register_best_model.py` errors about no runs** → You haven't trained anything yet. Run the
  experiments in Step 5 first.
- **`promote_model.py` says "No experiment named ... yet"** → You're either in the wrong folder,
  or you haven't trained that lab's runs yet. Make sure your prompt shows `chapter-04-mlflow`,
  then run that lab's training script first.
- **`Error: No such command 'set-registered-model-alias'`** → That command doesn't exist in
  MLflow. Nicknames can only be set from Python, which is exactly what `src/promote_model.py`
  does for you. Use the one-line command from section 4.9.4 instead.
- **A yellow `WARNING ... registering model based on models:/...` line appears** → Harmless.
  Newer MLflow versions store models in a slightly different spot, and this note just says it
  found yours. If the last three lines printed, it worked.

### Nuclear option — rebuild your toolbox from scratch

If your `venv` gets into a weird state (wrong Python version, half-installed packages, strange
import errors that won't go away), the cleanest fix is to throw the whole `venv` away and build
a fresh one. This turns off the broken environment, deletes it, builds a new one on **Python
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
> that version, install Python 3.12 first, or use whichever 3.11–3.12 version you have. Very new
> or very old Python versions sometimes trip up MLflow's dependencies, so pinning a known-good
> one avoids a whole class of install headaches.

## A real use case — the MLflow UI dashboard

Everything above records your runs. The **MLflow UI** is where that record pays off. Picture a
teammate asking: *"Which fraud model should we ship, and can you prove it's the best one?"*
Instead of digging through notes, you open the dashboard and answer in about a minute.

Start it from this chapter's folder (with `(venv)` active):

```bash
mlflow ui
```

Then open `http://127.0.0.1:5000` in your browser. Here's how you'd actually use it:

1. **Line up the contenders.** Click into the `fraud-detection-v1` experiment. Every run from
   Step 5 is a row, with its `max_depth`, `min_samples`, `criterion`, and the `accuracy`,
   `recall`, and `f1` scores in plain columns — a side-by-side view a hand-written log could
   never give you.
2. **Sort by what matters.** For fraud, catching real fraud matters most, so click the `recall`
   column to sort highest first. The winner floats to the top instantly.
3. **Compare head-to-head.** Tick the boxes on two or three runs and click **Compare**. MLflow
   lines up their settings and scores side by side, so you can see *exactly* which change moved
   the needle — for example, "going from `max_depth 3` to `8` lifted recall but barely touched
   accuracy."
4. **Prove it and hand it off.** Click the winning run to see its saved model, settings, and
   scores on one page. That page *is* your evidence. It's also the run you crown in Step 7 with
   `register_best_model.py`, so the model on the trophy shelf traces straight back to the numbers
   your teammate just saw.

> This is the everyday loop of a real ML engineer: train a few variants, open the dashboard, sort
> by the score that matters, compare, and register the winner — with a paper trail anyone can
> check. Press **Ctrl+C** in the terminal to stop the dashboard when you're done.

## 4.9–4.11 Three more jobs, the exact same trick

You've now trained a model, tracked it with MLflow, and crowned a winner once, on one problem.
The next three labs point that *exact same trick* at three different real-world jobs, so you can
see it's not a one-time party trick — it's how the whole industry works:

- **4.9 — a bank**, trying to catch fraud before it costs anyone money.
- **4.10 — a hospital**, trying to catch patients who might need to come back soon.
- **4.11 — a streaming service**, trying to catch subscribers who are about to cancel.

Each lab reuses the same MLflow setup from earlier in this chapter, but lives in its own new,
self-contained script under `usecases/`. The bank lab reuses `data/transactions.csv` from Step 4.
The hospital and streaming labs need two extra practice tables. Make them once, right now, from
this chapter's folder, before running either of those two labs:

```bash
python usecases/generate_usecase_data.py
```

**What you should see:** two lines telling you how many rows were made, and two new files land
in `data/`: `patient_encounters.csv` and `subscriber_activity.csv`.

## 4.9 Guided Lab: Catching Fraud at a Bank

*This lab assumes `mlflow ui` is already running at `127.0.0.1:5000`, exactly as you set up
earlier in this chapter. If you closed it, run `mlflow ui` again from Step 6.*

---

### 4.9.1 The Scenario

A bank's fraud team retrains its model all the time, because the tricks fraudsters use keep
changing. Their whole job comes down to one question, asked every single morning: **"Of all the
real fraud that happened yesterday, how much did we actually catch?"** That question has a
number attached to it — **recall** — and this whole lab is built around getting that number as
high as possible.

### 4.9.2 Script: `fintech_fraud_credit_lab.py`

*This file already exists at `usecases/fintech_fraud_credit_lab.py` — open it in VS Code to
follow along.*

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
    mlflow.sklearn.log_model(clf, name='fraud_model')
```

**Now let's read it in small bites.**

**STEP 1 — Load the transactions**

```python
df = pd.read_csv('data/transactions.csv')
X = df[['amount', 'time']]
y = df['is_fraud']
```

You're reading the same purchase table from earlier in this chapter. `X` holds the clues — the
amount and the time of day. `y` holds the answer key: was it fraud, yes or no.

**STEP 2 — Open a fresh notebook page**

```python
mlflow.set_experiment('fraud-detection-v1')
with mlflow.start_run():
```

This points MLflow at the same notebook section you already built earlier in this chapter. It's
just adding one more page to it, not starting over.

**STEP 3 — Write down your settings and grade the quiz**

```python
mlflow.log_param('max_depth', 6)
mlflow.log_metric('recall', recall_score(y_te, y_pred))
mlflow.sklearn.log_model(clf, name='fraud_model')
```

Nothing new here — you've done all three of these lines before. Run this script 4–5 times,
changing `max_depth` to 3, then 6, then 10, then 15 (edit the number right in the file each time,
in both the `mlflow.log_param` line and the `DecisionTreeClassifier` line, then rerun). Each run
adds a new row to the notebook. Without a few different runs to look at, the next part has
nothing to compare — so don't skip it.

---

### 4.9.3 Now Open the UI (this is the new part)

**Run this command**

```python
python usecases/fintech_fraud_credit_lab.py
```

**STEP 1 — Confirm you're in the right room**

Open your browser to `127.0.0.1:5000`. Look at the top of the sidebar. It must say **"Model
training,"** not "GenAI." If it says GenAI, click it to switch — you haven't broken anything,
you just walked in the wrong door.

**STEP 2 — Open the training runs table**

Click **Experiments → fraud-detection-v1 → Training runs.** You should see one row for every
time you ran the script — 4 or 5 rows, if you followed Step 3 above.

**STEP 3 — Choose what you can see**

Click **Columns**, near the top of the table. Turn on `max_depth`, `criterion`, `recall`,
`accuracy`, `f1`. Turn off anything you don't need — this table should be a workbench, not a
museum. Keep only what actually helps you decide.

**STEP 4 — Sort by the number that actually matters**

Click the **recall** column header once or twice, until you see **DESC** (biggest first). Your
best fraud-catcher is now sitting in row one, with zero math required from you.

**STEP 5 — See the pattern, not just the numbers**

Click the toggle near the top of the table that switches from **Table** to **Chart.** Choose a
scatter chart with `max_depth` on one axis and `recall` on the other. Look for the point where
the line bends downward — that's the run where the tree got too deep, started memorizing instead
of actually learning, and recall dropped. You just spotted **overfitting** with your own eyes,
no formula needed.

**STEP 6 — Ask the table a direct question**

In the **search bar** above the table, type:

```
metrics.recall > 0.85
```

Press enter. The table now shows only the runs a real compliance officer would even bother
looking at. This is the exact question a fraud team asks itself before trusting a model.

> **A Detour You Might Land On**
> If your search bar shows "No runs found" and you're sure some runs have good recall, check for
> a stray space or extra quote mark. The search box is picky — copy the line exactly as written.

**STEP 7 — Open the winner and look inside**

Click directly on the row-one run. Click the **Artifacts** tab. You'll see `fraud_model` sitting
there — the actual saved file, plus a manifest describing it. This is what "the model" really
is: not a mystery box, just a folder you can open and read.

---

### 4.9.4 Register and Promote

There are two jobs to do here, and one command does both:

1. **Register** — put the winning run on the trophy shelf under a name.
2. **Promote** — stick a nickname on it, like a gold medal sticker, so nobody has to remember a
   version number.

**STEP 1 — Run this one line** (from the `chapter-04-mlflow` folder, with `(venv)` showing):

```bash
python src/promote_model.py --experiment fraud-detection-v1 --metric recall --artifact fraud_model --model FraudDetectionModel --alias Champion
```

It's one long line. Copy the whole thing, paste it, press Enter.

**What you should see** (your score may differ — that's fine):

```
Best run: recall = 0.800
Registered FraudDetectionModel version 1
Nickname 'Champion' now points to version 1
```

**What the five flags mean, in plain words:**

| Flag | In plain words |
|---|---|
| `--experiment fraud-detection-v1` | Which group of runs to look in |
| `--metric recall` | Which score decides the winner |
| `--artifact fraud_model` | What the saved model is called inside each run |
| `--model FraudDetectionModel` | The name it gets on the trophy shelf |
| `--alias Champion` | The nickname to stick on the winner |

**STEP 2 — Go look at it.** In the web page, click **Models → FraudDetectionModel.** You'll see
the new version with a **Champion** badge next to it.

**What `promote_model.py` does, in plain words** (open it in VS Code to follow along):

1. `MlflowClient()` opens the same helper you used back in `register_best_model.py` — a tool
   that can reach into the notebook and pull runs back out.
2. `get_experiment_by_name(...)` finds the right notebook section — `fraud-detection-v1`.
3. `search_runs(...)` asks for every run in that section, sorted best-to-worst by recall.
4. `runs[0]` grabs the very first one in that sorted list — the winner.
5. `uri = f'runs:/{run_id}/fraud_model'` builds the winner's "home address" inside MLflow, so
   the next line knows exactly which saved model to grab.
6. `mlflow.register_model(...)` places that model on the trophy shelf under the name
   `FraudDetectionModel`.
7. `set_registered_model_alias(...)` sticks the `Champion` nickname on the version it just made.
   From now on, anyone who asks for "the Champion fraud model" gets exactly this one.

> **Why the nickname matters.** Every time you register a winner, MLflow makes a *new* version —
> 1, then 2, then 3. The nickname always moves to the newest one, so your other code can just
> ask for `Champion` and never needs updating. That's the whole trick.

Once a nickname exists, this is how you'd load that model anywhere else:

```python
model = mlflow.sklearn.load_model('models:/FraudDetectionModel@Champion')
```

The `@Champion` part is the payoff — real production code asks for the nickname, never a
version number.

---

### 4.9.5 Checkpoint: Can You...

- [ ] Explain, in one sentence, why recall matters more than accuracy for
      catching fraud?
- [ ] Sort the Training runs table by any metric, not just recall?
- [ ] Find the exact run behind a registered model version, using only the
      UI?

### 4.9.6 What Top FinTech Companies Track

Fraud teams retrain constantly because the tricks fraudsters use keep shifting — last week's
"best" model might not be this week's best. A real interview question you can now answer:
*"How would you know a fraud model needs retraining?"* — Answer: you'd watch recall drift down
across new runs, exactly the way you just watched it in this lab.

## 4.10 Guided Lab: Spotting Patients Who Might Come Back Too Soon

*Same tool, same web page, but a different kind of pressure — this time, a hospital. This lab
assumes `mlflow ui` is already running at `127.0.0.1:5000`.*

---

### 4.10.1 The Scenario

A hospital wants to know which patients are likely to be readmitted within 30 days, so care
teams can step in early and help before it happens again. The stakes here are different from the
bank: **every guess this model makes has to be explainable, a year from now, to an auditor or a
hospital review board.** MLflow's job in this lab isn't just remembering a score. It's building
a paper trail the hospital is required to keep.

The score this whole lab is built around is called **AUC** — how well the model ranks the
riskiest patients above the safest ones. Plain accuracy alone can hide the fact that a model is
quietly failing its sickest patients, so AUC is the fairer measuring stick here.

### 4.10.2 Script: `healthcare_readmission_lab.py`

*This file already exists at `usecases/healthcare_readmission_lab.py` — open it in VS Code to
follow along.*

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
    mlflow.sklearn.log_model(clf, name='readmission_model')
```

**Now let's read it in small bites.**

**STEP 1 — Load the patient encounters**

```python
df = pd.read_csv('data/patient_encounters.csv')
X = df[['age', 'length_of_stay', 'prior_admissions', 'chronic_conditions']]
y = df['readmitted_30d']
```

Same shape as every dataset in this chapter: `X` holds the clues (age, how many days they
stayed, how many times they've been admitted before, how many ongoing health conditions they
have), and `y` holds the answer you want predicted.

**STEP 2 — Stick a sticky note on the run before doing anything else**

```python
mlflow.set_tag('data_snapshot', '2026-Q2')
mlflow.set_tag('reviewed_by', 'pending')
```

This is new, and it's the single most important habit in this whole lab. A **tag** is a sticky
note MLflow attaches to a run, separate from its settings and scores. `data_snapshot` writes
down exactly which batch of patient data trained this model — because six months from now,
"which data trained this model?" is a question you *will* get asked, and "I think it was the
March data" is not good enough in a hospital.

**STEP 3 — Build the model and grade it a few different ways**

```python
mlflow.log_metric('auc', roc_auc_score(y_te, y_proba))
mlflow.log_metric('recall_high_risk', recall_score(y_te, y_pred))
```

`roc_auc_score` needs the model's *confidence* numbers (`y_proba`), not just its final yes/no
guesses. It's really asking "how well did the model rank risky patients above safe ones?"
instead of just "did it get the final answer right?" Run this script 3–4 times, editing
`n_estimators` (try 100, 200, 300) and `max_depth` (try 5, 8, 12) directly in the file each time.

---

### 4.10.3 Now Open the UI

**STEP 1 — Confirm the room**

`127.0.0.1:5000`, sidebar toggle on **"Model training."**

**STEP 2 — Open the table**

**Experiments → readmission-risk-v1 → Training runs.**

**STEP 3 — Turn on the columns that matter here**

Click **Columns**. This time, turn on `data_snapshot` and `reviewed_by` alongside `auc` and
`recall_high_risk`. In this lab, your sticky-note columns matter just as much as your score
columns — that's the healthcare-specific lesson.

**STEP 4 — Sort by AUC, not accuracy**

Click the **auc** column header to sort **DESC**. The run that's best at correctly ranking
high-risk patients rises to the top — a different sort than the bank lab, on purpose.

**STEP 5 — Ask a compliance question**

In the search bar, type:

```
tags.data_snapshot = '2026-Q2'
```

This shows you every run trained on one specific batch of data — exactly the question an
auditor would ask: "show me every model ever built on this data."

> **A Detour You Might Land On**
> Tag searches need quotes around the value. If you typed `tags.data_snapshot = 2026-Q2` without
> quotes, the search fails silently or errors out. The quotes are not optional.

**STEP 6 — Look at the chart view for a health check**

Switch to **Chart**, plot `n_estimators` against `auc`. If the line flattens out after 200,
that's your sign you've hit diminishing returns — more trees past that point cost more compute
without buying you any more accuracy.

**STEP 7 — Open the artifacts**

Click into the top run, open **Artifacts**, and confirm `readmission_model` is sitting there
with its full manifest. This is what you'd hand to a clinical reviewer alongside a prediction —
not just a score, the whole paper trail behind it.

---

### 4.10.4 Register — But Gate It Behind Human Review

> **Before you run this** — `promote_model.py` looks up an experiment that has to exist
> *already*. If you skipped straight here, you'll hit `No experiment named
> 'readmission-risk-v1' yet.` Make sure you've done these first, in order, with `(venv)` showing
> in your prompt:
>
> ```bash
> source venv/bin/activate                        # if (venv) isn't already showing
> python usecases/generate_usecase_data.py         # creates data/patient_encounters.csv
> python usecases/healthcare_readmission_lab.py    # trains runs, creates readmission-risk-v1
> ```

**STEP 1 — Run this one line** (from the `chapter-04-mlflow` folder):

```bash
python src/promote_model.py --experiment readmission-risk-v1 --metric auc --artifact readmission_model --model ReadmissionRiskModel --alias Staging
```

**What you should see:**

```
Best run: auc = 0.816
Registered ReadmissionRiskModel version 1
Nickname 'Staging' now points to version 1
```

It's the same script you used in the fraud lab, with three changes: the group of runs is
`readmission-risk-v1`, the score that picks the winner is `auc` instead of recall, and the
nickname is **Staging** instead of Champion.

**STEP 2 — Go look at it.** In the web page, click **Models → ReadmissionRiskModel.**

Notice that nickname says **Staging**, not Champion. In the **Models** tab you'll see this
version marked as staged — waiting on a clinician panel to review sample predictions before
anyone promotes it further. That pause, made visible right in the web page, is the whole point
of this lab: nothing in healthcare goes live just because a computer says "trust me."

Later, when the clinicians sign off, you promote **that exact version** — the one they actually
reviewed — by moving the nickname onto it. That's a shorter command, because there's no winner
to find and nothing new to register:

```bash
python src/promote_model.py --model ReadmissionRiskModel --alias Champion --version 1
```

**What you should see:**

```
Nickname 'Champion' now points to ReadmissionRiskModel version 1
```

This distinction matters in a hospital: the version a clinician approved is the version that
goes live, not a fresh one trained afterward.

---

### 4.10.5 Checkpoint: Can You...

- [ ] Explain why AUC matters more than accuracy for a readmission model?
- [ ] Find every run trained on a specific data snapshot, using only the
      search bar?
- [ ] Explain what a `Staging` alias means, versus a `Champion` alias?

### 4.10.6 What Top Healthcare AI Teams Track

Tagging every run with its exact data snapshot isn't extra polish — in regulated healthcare
work, it's often close to a requirement. A real interview question you can now answer: *"How
would you make sure a clinician signs off before a risk model goes live?"* — Answer: exactly the
alias-based staging gate you just built.

## 4.11 Guided Lab: Catching Subscribers About to Quit

*Third job, same exact habit — this time, a streaming service. This lab assumes `mlflow ui` is
already running at `127.0.0.1:5000`.*

---

### 4.11.1 The Scenario

A streaming platform — think Netflix, Peacock, any subscription service — wants to know which
subscribers are about to cancel, early enough to try to keep them around. Viewing habits shift
every week, so this model gets retrained all the time, just like the fraud model back in 4.9.
The score this lab is built around is **recall on the "about to cancel" group** — missing a
subscriber who's about to leave means losing them with zero chance to step in and help.

### 4.11.2 Script: `entertainment_churn_lab.py`

*This file already exists at `usecases/entertainment_churn_lab.py` — open it in VS Code to
follow along.*

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
    mlflow.sklearn.log_model(clf, name='churn_model')
```

**Now let's read it in small bites.**

**STEP 1 — Load subscriber activity**

```python
df = pd.read_csv('data/subscriber_activity.csv')
X = df[['avg_watch_hours_week', 'days_since_last_login',
        'titles_completed_30d', 'support_tickets_90d']]
y = df['churned_next_30d']
```

The clues this time: how much someone watches per week, how many days since they last logged
in, how many shows they've finished in the last month, and how many support tickets they filed.
The answer to predict: will they cancel in the next 30 days.

**STEP 2 — Open the experiment**

```python
mlflow.set_experiment('subscriber-churn-v1')
with mlflow.start_run():
```

A brand-new notebook section, kept completely separate from `fraud-detection-v1` and
`readmission-risk-v1` — every different business problem gets its own section, always.

**STEP 3 — Grade it and save it**

```python
mlflow.log_metric('recall_churn', recall_score(y_te, y_pred))
mlflow.sklearn.log_model(clf, name='churn_model')
```

Run this script 3–4 times, editing `n_estimators` (try 50, 150, 300) and `max_depth` (try 4, 7,
10) directly in the file each time.

---

### 4.11.3 Now Open the UI

**STEP 1 — Confirm the room**

`127.0.0.1:5000`, sidebar toggle on **"Model training."**

**STEP 2 — Open the table**

**Experiments → subscriber-churn-v1 → Training runs.**

**STEP 3 — Choose your columns**

Click **Columns**, turn on `n_estimators`, `max_depth`, `auc`, `recall_churn`.

**STEP 4 — Sort by the number that matters here**

Click **recall_churn** to sort **DESC** — same instinct as fraud back in 4.9: missing a real
about-to-quit subscriber costs more than sending one extra "please stay" email to someone who
was never leaving.

**STEP 5 — Find the diminishing-returns point**

Switch to **Chart**, plot `n_estimators` against `auc` as a scatter. Look for where the curve
goes flat — that's the point where adding more trees stops helping. A real platform team uses
this exact chart to decide how much compute a retraining run is actually worth paying for.

**STEP 6 — Compare two runs directly**

Select the checkboxes next to your top two runs in the table, then click **Compare.** MLflow
lines up their settings and scores side by side — this is how you'd explain to a teammate, in
one screenshot, exactly why one run beat the other.

> **A Detour You Might Land On**
> If Compare shows blank charts, make sure you selected at least two runs with the checkboxes
> first — Compare needs more than one run picked before it has anything to show.

**STEP 7 — Open the artifacts**

Click your top run, open **Artifacts**, confirm `churn_model` is there.

---

### 4.11.4 Register the Winner

> **Before you run this** — `promote_model.py` looks up an experiment that has to exist
> *already*. If you skipped straight here, you'll hit `No experiment named
> 'subscriber-churn-v1' yet.` Make sure you've done these first, in order, with `(venv)` showing
> in your prompt:
>
> ```bash
> source venv/bin/activate                        # if (venv) isn't already showing
> python usecases/generate_usecase_data.py         # creates data/subscriber_activity.csv
> python usecases/entertainment_churn_lab.py       # trains runs, creates subscriber-churn-v1
> ```

**STEP 1 — Run this one line** (from the `chapter-04-mlflow` folder):

```bash
python src/promote_model.py --experiment subscriber-churn-v1 --metric recall_churn --artifact churn_model --model ChurnRiskModel --alias Champion
```

**What you should see:**

```
Best run: recall_churn = 0.661
Registered ChurnRiskModel version 1
Nickname 'Champion' now points to version 1
```

Same script a third time, three changes: the runs live in `subscriber-churn-v1`, the score that
picks the winner is `recall_churn`, and the trophy-shelf name is `ChurnRiskModel`.

**STEP 2 — Go look at it.** In the web page, click **Models → ChurnRiskModel.** `Champion` now
points straight at this version, so anyone asking for "the Champion churn model" always gets
this exact one.

By now you've done this three times, in three different industries, with the same one-line
command. That repetition is the point — this is the habit, not a trick specific to any one job.

---

### 4.11.5 One Step Further: A Second System, Same Habit

Streaming platforms run a second model alongside churn — a **recommender**, deciding what to
show you next. You wouldn't train it the same way, but you'd log it into MLflow the exact same
way, just with different scores to track:

```python
mlflow.set_experiment('content-recommender-v1')
with mlflow.start_run():
    mlflow.log_param('embedding_dim', 64)
    mlflow.log_metric('precision_at_10', 0.34)  # of top 10 shown, % watched
    mlflow.log_metric('ndcg_at_10', 0.41)        # were the best ranked highest?
```

**What this code is doing, in plain words:** it opens a brand-new notebook section named
`content-recommender-v1`, writes down one setting (`embedding_dim`, which controls how the
recommender represents each show internally), then writes down two new kinds of scores instead
of recall or AUC — `precision_at_10` (out of the top 10 shows suggested, what fraction did the
person actually watch?) and `ndcg_at_10` (were the *very best* suggestions placed near the top of
the list, or buried?).

Back in the UI, you'd sort **Training runs** by `ndcg_at_10 DESC` — same click, same table,
completely different score. That's the real lesson of this whole chapter: **the habit never
changes. Only your judgment about what to track does.**

---

### 4.11.6 Checkpoint: Can You...

- [ ] Explain why recall matters for churn the same way it mattered for
      fraud?
- [ ] Use Compare to put two runs side by side?
- [ ] Name one metric a recommendation system would track that a churn model
      never would?

### 4.11.7 What Top Streaming Companies Track

Recommendation quality gets measured with precision@k, recall@k, and NDCG@k — not accuracy,
which doesn't even make sense for a ranked list of suggestions. A real interview question you
can now answer: *"Why wouldn't you use accuracy to evaluate a recommender?"* — Answer: accuracy
needs one single right answer to check against, but a recommendation is a whole ranked list, so
you need a score that rewards putting the best matches near the top. That's exactly what NDCG
does.

## What you just learned

- **MLflow** is a self-writing lab notebook — it does the **experiment tracking** for you.
- **Hyperparameters** are the settings you choose before training; MLflow records which ones you
  used.
- The **MLflow UI** lets you compare many runs side by side in your browser.
- A **model registry** is a trophy shelf for your best model, with a clear name and version.
- **Tags** are sticky notes on a run, and **aliases** (like Champion or Staging) are nicknames
  stuck on a model version — both make a run or model easy to find and trust later.
- The same notebook-and-trophy-shelf habit works on any problem — you just proved it three
  times, on a bank, a hospital, and a streaming service.
- This replaces the hand-written log file from Chapter 02 — same idea, far less effort and far
  fewer mistakes.

## Where to next

➡ [Chapter 05 — Let other programs talk to your model](../chapter-05-api). You'll wrap your fraud
model in a "front desk" so other programs can send it a purchase and get back an answer.
