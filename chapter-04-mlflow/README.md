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
