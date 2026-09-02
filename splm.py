#!/usr/bin/env python3

# Add to top of splm.py
"""SPLM - Core conversation model engine."""

# pylint: disable=too-many-instance-attributes, too-many-locals, too-many-arguments, too-many-positional-arguments, too-many-return-statements

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def detokenize(tokens: Iterable[str]) -> str:
    text = ""
    for token in tokens:
        if not text:
            text = token
        elif token in {".", ",", "!", "?", ";", ":", ")", "]", "}"}:
            text += token
        elif text[-1] in {"(", "[", "{"}:
            text += token
        else:
            text += " " + token
    return text


def animate_print(text: str, prefix: str = "bot: ", delay: float = 0.02) -> None:
    sys.stdout.write(prefix)
    sys.stdout.flush()
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()


class Vocabulary:
    def __init__(self, tokens: Iterable[str]):
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        self.itos = list(SPECIAL_TOKENS)
        for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            if token not in SPECIAL_TOKENS:
                self.itos.append(token)
        self.stoi = {token: index for index, token in enumerate(self.itos)}

    def encode(self, tokens: Iterable[str]) -> list[int]:
        unknown = self.stoi["<unk>"]
        return [self.stoi.get(token, unknown) for token in tokens]

    def decode(self, ids: Iterable[int]) -> list[str]:
        return [self.itos[index] if 0 <= index < len(self.itos) else "<unk>" for index in ids]

    def to_dict(self) -> dict:
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, data: dict) -> "Vocabulary":
        vocab = cls([])
        vocab.itos = list(data["itos"])
        vocab.stoi = {token: index for index, token in enumerate(vocab.itos)}
        return vocab


def softmax(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    exps = [math.exp(value - max_logit) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def tanh_vector(values: list[float]) -> list[float]:
    return [math.tanh(value) for value in values]


def zeros(rows: int, cols: int) -> list[list[float]]:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def random_matrix(rows: int, cols: int, scale: float) -> list[list[float]]:
    return [[random.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]


def random_vector(size: int, scale: float) -> list[float]:
    return [random.uniform(-scale, scale) for _ in range(size)]


@dataclass
class ForwardPass:
    x: list[float]
    hidden: list[float]
    logits: list[float]
    probs: list[float]


class TokenPredictor:
    def __init__(self, vocab_size: int, embed_dim: int = 8, hidden_dim: int = 16):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        scale = 0.1
        self.embeddings = random_matrix(vocab_size, embed_dim, scale)
        self.w1 = random_matrix(hidden_dim, embed_dim, scale)
        self.b1 = random_vector(hidden_dim, scale)
        self.w2 = random_matrix(vocab_size, hidden_dim, scale)
        self.b2 = random_vector(vocab_size, scale)

    def forward(self, context_ids: list[int]) -> ForwardPass:
        x = [0.0 for _ in range(self.embed_dim)]
        if context_ids:
            for token_id in context_ids:
                if token_id < len(self.embeddings):
                    embedding = self.embeddings[token_id]
                    for index, value in enumerate(embedding):
                        x[index] += value
            scale = 1.0 / len(context_ids)
            x = [value * scale for value in x]

        hidden_linear = []
        for row_index in range(self.hidden_dim):
            total = self.b1[row_index]
            row = self.w1[row_index]
            for col_index, value in enumerate(x):
                total += row[col_index] * value
            hidden_linear.append(total)
        hidden = tanh_vector(hidden_linear)

        logits = []
        for row_index in range(self.vocab_size):
            total = self.b2[row_index]
            row = self.w2[row_index]
            for col_index, value in enumerate(hidden):
                total += row[col_index] * value
            logits.append(total)

        probs = softmax(logits)
        return ForwardPass(x=x, hidden=hidden, logits=logits, probs=probs)

    def train_step(self, context_ids: list[int], target_id: int, learning_rate: float) -> float:
        pass_data = self.forward(context_ids)
        probs = pass_data.probs
        loss = -math.log(max(probs[target_id], 1e-12))

        grad_logits = probs[:]
        grad_logits[target_id] -= 1.0

        grad_hidden = [0.0 for _ in range(self.hidden_dim)]
        for row_index in range(self.vocab_size):
            grad = grad_logits[row_index]
            self.b2[row_index] -= learning_rate * grad
            row = self.w2[row_index]
            for col_index in range(self.hidden_dim):
                grad_hidden[col_index] += row[col_index] * grad
                row[col_index] -= learning_rate * grad * pass_data.hidden[col_index]

        grad_hidden_linear = [grad_hidden[index] * (1.0 - pass_data.hidden[index] ** 2) for index in range(self.hidden_dim)]

        grad_x = [0.0 for _ in range(self.embed_dim)]
        for row_index in range(self.hidden_dim):
            grad = grad_hidden_linear[row_index]
            self.b1[row_index] -= learning_rate * grad
            row = self.w1[row_index]
            for col_index in range(self.embed_dim):
                grad_x[col_index] += row[col_index] * grad
                row[col_index] -= learning_rate * grad * pass_data.x[col_index]

        if context_ids:
            scale = learning_rate / len(context_ids)
            for token_id in context_ids:
                if token_id < len(self.embeddings):
                    embedding = self.embeddings[token_id]
                    for index in range(self.embed_dim):
                        embedding[index] -= scale * grad_x[index]

        return loss

    def predict_next(self, context_ids: list[int], top_k: int | None = None, temperature: float = 1.0) -> int:
        probs = self.forward(context_ids).probs
        if temperature <= 0:
            return max(range(len(probs)), key=probs.__getitem__)

        adjusted = [math.log(max(prob, 1e-12)) / temperature for prob in probs]
        sampled = softmax(adjusted)
        candidates = list(range(len(sampled)))
        if top_k is not None and 0 < top_k < len(candidates):
            top_indices = sorted(candidates, key=lambda index: sampled[index], reverse=True)[:top_k]
            probabilities = [sampled[index] for index in top_indices]
            total = sum(probabilities)
            probabilities = [probability / total for probability in probabilities]
            return random.choices(top_indices, weights=probabilities, k=1)[0]
        return random.choices(candidates, weights=sampled, k=1)[0]

    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "embeddings": self.embeddings,
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TokenPredictor":
        model = cls(data["vocab_size"], data["embed_dim"], data["hidden_dim"])
        model.embeddings = data["embeddings"]
        model.w1 = data["w1"]
        model.b1 = data["b1"]
        model.w2 = data["w2"]
        model.b2 = data["b2"]
        return model


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


ROLE_ORDER = ["interjection", "pronoun", "verb", "adjective", "noun", "adverb", "preposition", "determiner", "conjunction"]
ROLE_SEEDS = {
    "interjection": {"hey", "yeah", "sure", "ok", "okay", "right", "hmm"},
    "pronoun": {"i", "you", "we", "they", "it", "he", "she", "me", "us", "them"},
    "verb": {"am", "is", "are", "be", "was", "were", "do", "does", "did", "have", "has", "had", "keep", "keeps", "move", "moves", "stay", "stays", "make", "makes", "say", "says", "try", "tries", "help", "helps", "want", "wants", "think", "thinks", "feel", "feels", "learn", "learns", "respond", "responds", "start", "starts", "cut", "cuts", "pick", "picks", "repeat", "repeats", "avoid", "avoids", "give", "gives", "talk", "talks", "trim", "trims", "sort", "sorts", "send", "sends", "listen", "listens", "remember", "remembers", "reach", "reaches", "change", "changes", "pick", "picks", "move", "moves", "hold", "holds", "wait", "waits", "use", "uses", "go", "goes", "tell", "tells"},
    "adjective": {"steady", "clear", "simple", "calm", "quiet", "direct", "useful", "clean", "plain", "honest", "focused", "practical", "warm", "small", "ready", "busy", "steady", "worthwhile", "manageable"},
    "noun": {"thing", "details", "message", "step", "tone", "pattern", "thread", "plan", "path", "idea", "answer", "reply", "pace", "time", "room", "noise", "hint", "part", "work", "shift"},
    "adverb": {"now", "then", "just", "still", "again", "slowly", "quickly", "soon", "here", "there", "today", "tonight"},
    "preposition": {"to", "from", "with", "for", "in", "on", "at", "by", "of", "into", "over", "under", "about", "after", "before", "through"},
    "determiner": {"a", "an", "the", "this", "that", "these", "those", "my", "your", "our", "their"},
    "conjunction": {"and", "or", "but", "so", "yet", "while", "because", "if", "when"},
}
DEFAULT_ROLE_WORDS = {
    "interjection": "yeah",
    "pronoun": "i",
    "verb": "stay",
    "adjective": "steady",
    "noun": "thing",
    "adverb": "now",
    "preposition": "with",
    "determiner": "the",
    "conjunction": "and",
}
STOP_WORDS = {"a", "an", "the", "and", "or", "but", "so", "yet", "if", "when", "while", "to", "of", "in", "on", "at", "for", "from", "with", "by", "about", "into", "through", "after", "before"}


def normalize_word(token: str) -> str:
    return re.sub(r"^[^a-z0-9']+|[^a-z0-9']+$", "", token.lower())


def word_tokens(text: str) -> list[str]:
    return [token for token in (normalize_word(piece) for piece in tokenize(text)) if token]


def classify_word(word: str) -> str:
    if word in ROLE_SEEDS["interjection"]:
        return "interjection"
    if word in ROLE_SEEDS["pronoun"]:
        return "pronoun"
    if word in ROLE_SEEDS["verb"] or word.endswith(("ing", "ed", "en")):
        return "verb"
    if word in ROLE_SEEDS["adjective"] or word.endswith(("ous", "ful", "less", "ive", "al", "ic", "ish", "ary", "ory", "able", "ible", "ant", "ent", "y")):
        return "adjective"
    if word in ROLE_SEEDS["adverb"] or word.endswith("ly"):
        return "adverb"
    if word in ROLE_SEEDS["preposition"]:
        return "preposition"
    if word in ROLE_SEEDS["determiner"]:
        return "determiner"
    if word in ROLE_SEEDS["conjunction"]:
        return "conjunction"
    return "noun"


def tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(tokens)
    total = sum(counts.values()) or 1
    return {token: (count / total) * idf.get(token, 1.0) for token, count in counts.items()}


def cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0

    dot = sum(weight * right.get(token, 0.0) for token, weight in left.items())
    left_norm = math.sqrt(sum(weight * weight for weight in left.values()))
    right_norm = math.sqrt(sum(weight * weight for weight in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


@dataclass
class SearchHit:
    score: float
    prompt: str
    response: str


class LocalMarkovGenerator:
    def __init__(self, n: int = 2):
        self.n = n
        self.transitions: dict[tuple[str, ...], dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def fit_weighted(self, matches: list[SearchHit]) -> None:
        self.transitions.clear()
        for hit in matches:
            weight = max(hit.score, 0.01)
            tokens = word_tokens(hit.response)
            for i in range(len(tokens) - self.n):
                state = tuple(tokens[i : i + self.n])
                next_word = tokens[i + self.n]
                self.transitions[state][next_word] += weight

    def get_candidate_weights(self, state: tuple[str, ...]) -> dict[str, float]:
        return self.transitions.get(state, {})


class ConversationModel:
    def __init__(
        self,
        prompts: list[str],
        responses: list[str],
        prompt_vectors: list[dict[str, float]],
        idf: dict[str, float],
        word_roles: dict[str, str],
        vocab: list[str],
        predictor: TokenPredictor | None = None,
    ):
        self.prompts = prompts
        self.responses = responses
        self.prompt_vectors = prompt_vectors
        self.idf = idf
        self.word_roles = word_roles
        self.vocab = vocab
        self.vocab_obj = Vocabulary(vocab)
        self.predictor = predictor or TokenPredictor(vocab_size=len(self.vocab_obj.itos))
        self.markov = LocalMarkovGenerator(n=2)

    @classmethod
    def from_files(cls, prompts_path: Path, responses_path: Path, log_every: int = 5) -> "ConversationModel":
        prompts = [line.strip() for line in prompts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        responses = [line.strip() for line in responses_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not prompts or not responses:
            raise ValueError("training files must not be empty")
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must have the same number of non-empty lines")

        print(f"training on {len(prompts)} prompt/response pairs", flush=True)
        prompt_tokens = []
        doc_frequency: Counter[str] = Counter()
        role_counts: dict[str, Counter[str]] = {}
        vocab_tokens: set[str] = set()

        for index, (prompt, response) in enumerate(zip(prompts, responses), start=1):
            tokens = word_tokens(prompt)
            prompt_tokens.append(tokens)
            for token in set(tokens):
                doc_frequency[token] += 1
            vocab_tokens.update(tokens)

            for token in word_tokens(response):
                vocab_tokens.add(token)
                role = classify_word(token)
                counts = role_counts.setdefault(token, Counter())
                counts[role] += 1

            if index % log_every == 0 or index == len(prompts):
                print(f"indexed {index}/{len(prompts)} pairs", flush=True)

        idf = {token: math.log((1 + len(prompts)) / (1 + frequency)) + 1.0 for token, frequency in doc_frequency.items()}
        prompt_vectors = [tfidf_vector(tokens, idf) for tokens in prompt_tokens]
        word_roles = {token: counts.most_common(1)[0][0] for token, counts in role_counts.items()}
        vocab = sorted(vocab_tokens)

        vocab_obj = Vocabulary(vocab)
        predictor = TokenPredictor(vocab_size=len(vocab_obj.itos))

        print("built prompt scorer, word-role dictionary, and neural predictor", flush=True)
        return cls(
            prompts=prompts,
            responses=responses,
            prompt_vectors=prompt_vectors,
            idf=idf,
            word_roles=word_roles,
            vocab=vocab,
            predictor=predictor,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationModel":
        vocab = list(data.get("vocab", sorted({token for prompt in data["prompts"] for token in word_tokens(prompt)} | {token for response in data["responses"] for token in word_tokens(response)})))
        predictor = TokenPredictor.from_dict(data["predictor"]) if "predictor" in data else None
        return cls(
            prompts=list(data["prompts"]),
            responses=list(data["responses"]),
            prompt_vectors=[dict(vector) for vector in data["prompt_vectors"]],
            idf={token: float(value) for token, value in data["idf"].items()},
            word_roles={token: str(role) for token, role in data["word_roles"].items()},
            vocab=vocab,
            predictor=predictor,
        )

    def to_dict(self) -> dict:
        return {
            "prompts": self.prompts,
            "responses": self.responses,
            "prompt_vectors": self.prompt_vectors,
            "idf": self.idf,
            "word_roles": self.word_roles,
            "vocab": self.vocab,
            "predictor": self.predictor.to_dict(),
        }

    def _debug_vector_summary(self, vector: dict[str, float], limit: int = 10) -> str:
        if not vector:
            return "(empty)"
        items = sorted(vector.items(), key=lambda item: (-item[1], item[0]))[:limit]
        return ", ".join(f"{token}:{weight:.3f}" for token, weight in items)

    def _debug_role_summary(self, role_counters: dict[str, Counter[str]], limit: int = 5) -> list[str]:
        lines: list[str] = []
        for role in ROLE_ORDER:
            counter = role_counters.get(role)
            if not counter:
                continue
            top_tokens = ", ".join(f"{token}:{weight:.2f}" for token, weight in counter.most_common(limit))
            lines.append(f"{role}: {top_tokens}")
        return lines

    def find_matches(self, query: str, top_n: int = 5) -> list[SearchHit]:
        query_tokens = word_tokens(query)
        query_vector = tfidf_vector(query_tokens, self.idf)
        matches = [
            SearchHit(score=cosine_similarity(query_vector, prompt_vector), prompt=prompt, response=response)
            for prompt, response, prompt_vector in zip(self.prompts, self.responses, self.prompt_vectors)
        ]
        matches = [match for match in matches if match.score > 0.0]
        matches.sort(key=lambda match: match.score, reverse=True)
        if not matches:
            return [SearchHit(score=0.0, prompt=self.prompts[0], response=self.responses[0])]
        return matches[:top_n]

    def respond(self, query: str, max_words: int = 40, show_matches: bool = False, debug: bool = False) -> str:
        matches = self.find_matches(query, top_n=5)

        self.markov.fit_weighted(matches)

        query_words = word_tokens(query)
        context_ids = self.vocab_obj.encode(query_words)

        seed_state = None
        for i in range(len(query_words) - self.markov.n + 1):
            state_candidate = tuple(query_words[i : i + self.markov.n])
            if state_candidate in self.markov.transitions:
                seed_state = state_candidate
                break

        if not seed_state and self.markov.transitions:
            seed_state = random.choice(list(self.markov.transitions.keys()))

        if not seed_state:
            return matches[0].response

        generated_tokens = list(seed_state)
        current_state = seed_state

        for _ in range(max_words - len(seed_state)):
            candidate_weights = self.markov.get_candidate_weights(current_state)
            if not candidate_weights:
                break

            pass_data = self.predictor.forward(context_ids)

            best_word = None
            best_score = -1.0

            for word, weight in candidate_weights.items():
                word_id = self.vocab_obj.encode([word])[0]
                neural_prob = pass_data.probs[word_id] if word_id < len(pass_data.probs) else 1e-12
                combined_score = weight * neural_prob

                if combined_score > best_score:
                    best_score = combined_score
                    best_word = word

            next_word = best_word or list(candidate_weights.keys())[0]
            generated_tokens.append(next_word)
            next_word_id = self.vocab_obj.encode([next_word])[0]
            context_ids.append(next_word_id)
            current_state = tuple((list(current_state) + [next_word])[-self.markov.n:])

            if next_word in {".", "!", "?"}:
                break

        output = detokenize(generated_tokens)

        if show_matches:
            lines = [f"match {index + 1}: {hit.score:.3f} | {hit.prompt}" for index, hit in enumerate(matches)]
            print("\n".join(lines), flush=True)

        return output


def train_model(prompts_path: Path, responses_path: Path, output_path: Path, log_every: int) -> None:
    random.seed(7)
    model = ConversationModel.from_files(prompts_path, responses_path, log_every=log_every)
    output_path.write_text(json.dumps(model.to_dict(), ensure_ascii=True, indent=2), encoding="utf-8")


def load_model(model_path: Path) -> ConversationModel:
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    return ConversationModel.from_dict(payload)


def generate_text(model: ConversationModel, prompt: str, max_tokens: int, show_matches: bool, debug: bool) -> str:
    return model.respond(prompt, max_words=max_tokens, show_matches=show_matches, debug=debug)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Call-and-response conversational chatbot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model from paired prompt and response files")
    train_parser.add_argument("--prompts", type=Path, default=Path("prompts.txt"))
    train_parser.add_argument("--responses", type=Path, default=Path("responses.txt"))
    train_parser.add_argument("--output", type=Path, default=Path("model.json"))
    train_parser.add_argument("--log-every", type=int, default=5)

    chat_parser = subparsers.add_parser("chat", help="Chat with a trained model")
    chat_parser.add_argument("--model", type=Path, default=Path("model.json"))
    chat_parser.add_argument("--prompt", type=str, default="hello")
    chat_parser.add_argument("--max-tokens", type=int, default=40)
    chat_parser.add_argument("--interactive", action="store_true")
    chat_parser.add_argument("--show-matches", action="store_true")
    chat_parser.add_argument("--debug", action="store_true", help="Print prompt scoring, role buckets, and response assembly")

    return parser


def run_chat(model_path: Path, prompt: str, max_tokens: int, interactive: bool, show_matches: bool, debug: bool) -> None:
    model = load_model(model_path)
    if interactive:
        print("type a message and press return. empty line exits.")
        while True:
            prompt = input("you: ").strip()
            if not prompt:
                break
            response = generate_text(model, prompt, max_tokens, show_matches, debug)
            animate_print(response)
        return

    response = generate_text(model, prompt, max_tokens, show_matches, debug)
    animate_print(response)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "train":
        train_model(
            prompts_path=args.prompts,
            responses_path=args.responses,
            output_path=args.output,
            log_every=args.log_every,
        )
    elif args.command == "chat":
        run_chat(args.model, args.prompt, args.max_tokens, args.interactive, args.show_matches, args.debug)


if __name__ == "__main__":
    main()