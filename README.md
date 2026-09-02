# Single-Pass Language Model (SPLM)

<p align="center">
  <a href="https://github.com/npmInstallSnack/SPLM"><img alt="Repo stars" src="https://img.shields.io/github/stars/npmInstallSnack/SPLM?style=flat"></a>
  <a href="https://github.com/prettier/prettier"><img alt="code style: prettier" src="https://img.shields.io/badge/code_style-prettier-ff69b4.svg" /></a>
  <a href="https://github.com/npmInstallSnack/SPLM/pulls"><img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" /></a>
  <a href="https://github.com/npmInstallSnack/SPLM/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/npmInstallSnack/SPLM" /></a>
  <a href="https://github.com/npmInstallSnack/SPLM"><img alt="Size" src="https://img.shields.io/github/repo-size/npmInstallSnack/SPLM" /></a>
</p>

A lightweight, zero-dependency Python chatbot that combines TF-IDF prompt matching, a dynamic local Markov chain, and a neural token predictor to generate conversational responses while requiring only one pass of training. It also includes a browser-based dark-mode interface for a cleaner chat experience.

---

**Prerequisites**

- **Python 3.8+** (Uses standard library modules only—no `pip install` required).

---

**1. Custom Data Setup**

Create two plain text files in your project directory: `prompts.txt` and `responses.txt`. Each line in `prompts.txt` must correspond directly to the same line number in `responses.txt`. Two template files are provided.

**`prompts.txt`**

```text
hello
how are you
what is your name
what do you do

```

**`responses.txt`**

```text
hey there! how can I help you today?
i am doing great and ready to work.
my name is SPLM, nice to meet you.
i process text and generate contextual answers.

```

---

**2. Training the Model**

Run the `train` command to process your prompt/response pairs, build the TF-IDF vocabulary, and save the initialized model weights to a JSON file.

**Basic Training**

```bash
python3 splm.py train

```

**Custom File Paths**

```bash
python3 splm.py train --prompts my_prompts.txt --responses my_responses.txt --output my_model.json

```

| Argument      | Default         | Description                                     |
| ------------- | --------------- | ----------------------------------------------- |
| `--prompts`   | `prompts.txt`   | Path to the prompt dataset file                 |
| `--responses` | `responses.txt` | Path to the response dataset file               |
| `--output`    | `model.json`    | Path where the trained model JSON will be saved |
| `--log-every` | `5`             | Progress print interval during indexing         |

---

**3. Chatting with the Bot**

Use the `chat` command to interact with your trained model.

**Single Prompt Mode**

```bash
python3 splm.py chat --prompt "hello how are you"

```

**Interactive Terminal Mode**
Launch a continuous chat loop in your console:

```bash
python3 splm.py chat --interactive

```

_(Press `Return` on an empty line to exit interactive mode.)_

**Advanced Options & Debugging**

- **Show Matched Prompts:** Displays the top 5 TF-IDF matches and their similarity scores before generating an answer.

```bash
python3 splm.py chat --prompt "tell me a plan" --show-matches

```

- **Enable Debugging Logs:** Prints internal query vectors, top matches, and generation steps.

```bash
python3 splm.py chat --prompt "hello" --debug

```

- **Limit Max Tokens:** Control maximum response length.

```bash
python3 splm.py chat --interactive --max-tokens 20

```

| Argument         | Default      | Description                                       |
| ---------------- | ------------ | ------------------------------------------------- |
| `--model`        | `model.json` | Path to the trained model JSON file               |
| `--prompt`       | `"hello"`    | Input query (used in single-prompt mode)          |
| `--max-tokens`   | `40`         | Maximum number of tokens to generate              |
| `--interactive`  | `False`      | Launches interactive chat session                 |
| `--show-matches` | `False`      | Prints top retrieved responses and TF-IDF scores  |
| `--debug`        | `False`      | Prints verbose scoring and token prediction steps |

---

**4. Web Interface**

Launch the browser UI with a local web server:

```bash
python3 web_chat.py --model model.json --open
```

The web interface uses a modern dark layout, message bubbles, a live input composer, and an optional match panel that shows the closest prompts used to build each reply.

SPLM (c) 2026 by npmInstallSnack
