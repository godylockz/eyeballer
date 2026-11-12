# Eyeballer

![Logo](/docs/eyeballer_logo.png)


Give those screenshots of yours a quick eyeballing.

Eyeballer is meant for large-scope network penetration tests where you need to find "interesting" targets from a huge set of web-based hosts. Go ahead and use your favorite screenshotting tool like normal (EyeWitness or GoWitness) and then run them through Eyeballer to tell you what's likely to contain vulnerabilities, and what isn't.

Give it a try live at: https://eyeballer.bishopfox.com

### Example Labels

| Old-Looking Sites | Login Pages |
| ------ |:-----:|
| ![Sample Old-looking Page](/docs/old-looking.png) | ![Sample Login Page](/docs/login.png) |

| Webapp | Custom 404's |
| ------ |:-----:|
| ![Sample Webapp](/docs/homepage.png) | ![Sample Custom 404](/docs/404.png) |

| Parked Domains |
| ------ |
| ![Sample Webapp](/docs/parked.png) |

## What the Labels Mean

**Old-Looking Sites**
Blocky frames, broken CSS, that certain "je ne sais quoi" of a website that looks like it was designed in the early 2000's. You know it when you see it. Old websites aren't just ugly, they're also typically super vulnerable. When you're looking to hack into something, these websites are a gold mine.

**Login Pages**
Login pages are valuable to pen testing, they indicate that there's additional functionality you don't currently have access to. It also means there's a simple follow-up process of credential enumeration attacks. You might think that you can set a simple heuristic to find login pages, but in practice it's really hard. Modern sites don't just use a simple input tag we can grep for.

**Webapp**
This tells you that there is a larger group of pages and functionality available here that can serve as surface area to attack. This is in contrast to a simple login page, with no other functionality. Or a default IIS landing page which has no other functionality. This label should indicate to you that there is a web application here to attack.

**Custom 404**
Modern sites love to have cutesy custom 404 pages with pictures of broken robots or sad looking dogs. Unfortunately, they also love to return HTTP 200 response codes while they do it. More often, the "404" page doesn't even contain the text "404" in it. These pages are typically uninteresting, despite having a lot going on visually, and Eyeballer can help you sift them out.

**Parked Domains**
Parked domains are websites that look real, but aren't valid attack surface. They're stand-in pages, usually devoid of any real functionality, consist almost entirely of ads, and are usually not run by our actual target. It's what you get when the domain specified is wrong or lapsed. Finding these pages and removing them from scope is really valuable over time.

## Setup

Eyeballer needs CPython 3.10 through 3.13 (TensorFlow 2.21 has no newer
wheels). `requirements.lock` pins every runtime dependency and its hashes.

Download the v3 pretrained weights from the [upstream release](https://github.com/BishopFox/eyeballer/releases/tag/3.0)
into the package directory. Installs bundle them, so `predict` works offline
without `--weights`:

```sh
curl --proto '=https' --tlsv1.2 -fL -o eyeballer/bishop-fox-pretrained-v3.h5 \
  https://github.com/BishopFox/eyeballer/releases/download/3.0/bishop-fox-pretrained-v3.h5
printf '%s  %s\n' dd1c3741f6141f096a5ca6ee882df74bd5cb5b907c2df4b5548c4b02bd42afcd \
  eyeballer/bishop-fox-pretrained-v3.h5 | sha256sum --check
```

Install the `eyeballer` command with pipx, pinned to the lock's versions. Hashed
constraints switch pip into hash-checking mode, which rejects the local
directory, so pass version-only constraints:

```sh
sed -nE 's/^([A-Za-z0-9][A-Za-z0-9._-]*==[^ ]+).*/\1/p' requirements.lock >constraints.txt
pipx install . --python 3.12 --fetch-python=missing --backend pip \
  --pip-args="--constraint=$PWD/constraints.txt"
```

For development, use a Python 3.12 virtual environment:

```sh
python3.12 -m venv .venv
.venv/bin/pip install --require-hashes -r requirements.lock
.venv/bin/pip install --no-deps --editable .
```

The tested install uses CPU; CUDA setup is outside this workflow. To refresh
the lock after checking dependency advisories:

```sh
uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 \
  --generate-hashes --output-file requirements.lock
```

**Training Data** You can find our training data here:

https://www.kaggle.com/altf42600/pentest-screensots

There's two things you need from the training data:

1. `images/` folder, containing all the screenshots (resized down to 224x224)
2. `labels.csv` that has all the labels

Copy both into the root of the Eyeballer code tree.

Additionally, you can find a pretrained weights file you can use right out of the box without training.  

1. `bishop-fox-pretrained-vN.h5`

It's here on GitHub, look at the `releases` section for the latest one.

## Predicting Labels

NOTE: For best results, make sure you screenshot your websites in a native 1.6x aspect ratio. IE: 1440x900. Eyeballer will scale the image down automatically to the right size for you, but if it's the wrong aspect ratio then it will squish in a way that will affect prediction performance.

To eyeball some screenshots, just run the "predict" mode:

```
eyeballer predict YOUR_FILE.png
```

Pass `--weights YOUR_WEIGHTS.h5` before `predict` to use other weights. For a whole directory of files:

```
eyeballer predict PATH_TO/YOUR_FILES/ --output-dir ./eyeballer
```

Eyeballer will spit the results back to you in human readable format (a `results.html` file so you can browse it easily) and machine readable format (a `results.csv` file).

Both single-image and directory runs write `results.csv` and `results.html`.
CSV filenames are absolute local paths; the HTML report uses local file links.
Missing weights or an input with no readable images exits nonzero without
writing new reports. Directory traversal is non-recursive. Keep each scan's
reports with its screenshots; re-run predictions if the screenshot tree moves.

For gowitness, classify its `shots/` or `screenshots/` directory after capture.
Keep the gowitness database or JSONL export: image names alone do not reliably
identify URLs. Join CSV `filename` to gowitness `file_name` (JSON) or `filename`
(SQLite), resolved against the scan directory. Scores are independent values in
[0, 1]; the default label cutoff is strictly greater than 0.5. These predictions
help prioritize manual review and do not establish vulnerabilities or scope.

## Tests

```sh
EYEBALLER_TEST_WEIGHTS="$PWD/eyeballer/bishop-fox-pretrained-v3.h5" \
  .venv/bin/python -m unittest discover -s tests -v
```

CLI tests run without loading TensorFlow. Model tests use the explicitly supplied,
checksum-verified v3 weights; without that variable they are reported as skipped.
The old four-label test model is incompatible with the five-label network.

## Performance

Eyeballer's performance is measured against an evaluation dataset, which is 20% of the overall screenshots chosen at random. Since these screenshots are never used in training, they can be an effective way to see how well the model is performing. Here are the latest results:

| Overall Binary Accuracy | 93.52% |
| ------ |:-----:|
| **All-or-Nothing Accuracy** | **76.09%** |

**Overall Binary Accuracy** is probably what you think of as the model's "accuracy". It's the chance, given any single label, that it is correct.

**All-or-Nothing Accuracy** is more strict. For this, we consider all of an image's labels and consider it a failure if ANY label is wrong. This accuracy rating is the chance that the model correctly predicts all labels for any given image.

| Label | Precision | Recall |
| ------ | ------ |:-----:|
| Custom 404 | 80.20% | 91.01% |
| Login Page | 86.41% | 88.47% |
| Webapp | 95.32% | 96.83% |
| Old Looking | 91.70% | 62.20% |
| Parked Domain | 70.99% | 66.43% |

For a detailed explanation on [Precision vs Recall, check out Wikipedia](https://en.wikipedia.org/wiki/Precision_and_recall).

## Training
To train a new model, run:
```
eyeballer train
```

You'll want a machine with a good GPU for this to run in a reasonable amount of time. Setting that up is outside the scope of this readme, however.

This will output a new model file (weights.h5 by default).

## Evaluation

You just trained a new model, cool! Let's see how well it performs against some images it's never seen before, across a variety of metrics:

```
eyeballer --weights YOUR_WEIGHTS.h5 evaluate
```

The output will describe the model's accuracy in both recall and precision for each of the program's labels. (Including "none of the above" as a pseudo-label)
