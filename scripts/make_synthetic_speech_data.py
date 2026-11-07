"""Generate datasets/speech/SYNTHETIC_demo_transcripts.csv

IMPORTANT - this is NOT patient data. Every transcript is produced by the
templates below. The file exists so that the speech pipeline can be run,
demonstrated and tested before access to DementiaBank has been granted.

How it works: each transcript describes one of three everyday picture scenes.
A hidden "impairment" number between 0 and 1 is drawn for each speaker
(low for Control, higher for Dementia, with deliberate overlap). The higher
it is, the more details are left out, the more specific words are swapped for
vague ones, and the more fillers, repetitions and "I don't know" phrases appear.
"""
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

# Each fact: (specific wordings, vague wordings)
SCENES = {
    "kitchen": [
        (["a boy is standing on a stool", "the boy has climbed up on a stool", "there's a little boy up on a stool"],
         ["he is up on that thing", "somebody is up there", "the the boy is on something"]),
        (["he is reaching for the cookie jar", "he's taking cookies out of the jar", "he is stealing cookies from the cupboard"],
         ["he is getting something up there", "he's reaching for that", "taking the things out"]),
        (["the stool is tipping over", "the stool is about to fall", "he is going to fall off the stool"],
         ["it is going to fall", "that thing is going over", "he's falling I think"]),
        (["his sister is holding out her hand for a cookie", "the girl is asking him for a cookie", "the little girl is laughing with her finger to her lips"],
         ["she wants something", "the girl is there too", "she is doing something with her hand"]),
        (["the mother is washing dishes at the sink", "the mother is drying a plate", "their mother is standing at the sink with a dish towel"],
         ["the lady is doing the things", "she is washing something", "the woman is there at the the place"]),
        (["the sink is overflowing", "water is running over onto the floor", "the tap is on and the water is spilling over the sink"],
         ["the water is going", "it is all coming out", "there is water there"]),
        (["she is not noticing the water", "the mother is not paying attention to the children", "she seems to be daydreaming"],
         ["she doesn't see it", "she is not looking", "she doesn't know"]),
        (["there are two cups and a plate on the counter", "some dishes are sitting on the counter"],
         ["there are things on there", "some stuff there"]),
        (["through the window you can see the garden", "the window is open and there are curtains", "outside the window there is a path and some bushes"],
         ["there is a window", "you can see out there", "something outside"]),
    ],
    "park": [
        (["a family is having a picnic in the park", "a family is sitting on a blanket having a picnic"],
         ["some people are sitting there", "they are outside doing something"]),
        (["the father is reading a book under a tree", "the man is leaning against a tree with a book"],
         ["the man is doing something there", "he is sitting with a thing"]),
        (["the mother is pouring a drink", "the woman is pouring juice into a cup"],
         ["she is doing the the drink thing", "the lady has something"]),
        (["a boy is flying a kite", "the boy is running with a kite"],
         ["he is playing with that thing up there", "the boy has something in the air"]),
        (["a dog is chasing him", "the dog is running after the boy"],
         ["there is an animal", "the the dog is there"]),
        (["a girl is building a sandcastle near the water", "the girl is playing in the sand with a bucket"],
         ["she is playing with stuff", "the girl is down there"]),
        (["a sailboat is out on the lake", "there is a boat sailing on the lake"],
         ["there is something on the water", "a thing out there"]),
        (["a man is fishing from the dock", "someone is fishing off the pier"],
         ["somebody is there doing something", "he is standing there"]),
        (["the radio is playing next to the picnic basket", "there is a picnic basket and a radio on the blanket"],
         ["there are things on the the ground", "some stuff there"]),
    ],
    "street": [
        (["a woman is crossing the street with a shopping bag", "a lady is carrying groceries across the road"],
         ["she is walking with something", "the lady is going there"]),
        (["a car has stopped at the traffic light", "the car is waiting at the red light"],
         ["there is a car", "the thing stopped there"]),
        (["a cyclist is riding past the bakery", "a man on a bicycle is passing the bakery"],
         ["somebody is on a thing", "he is going by"]),
        (["a boy has dropped his ice cream on the pavement", "the little boy is crying because his ice cream fell"],
         ["he dropped something", "the boy is upset about that"]),
        (["a postman is putting letters in the mailbox", "the postman is delivering mail to the house"],
         ["the man is putting things in there", "he has stuff with him"]),
        (["a cat is sitting on the wall watching a bird", "there is a cat on the fence looking at a bird"],
         ["an animal is up there", "there is a thing on the the wall"]),
        (["the shopkeeper is sweeping in front of his shop", "a man is sweeping the pavement outside the store"],
         ["he is doing something with that", "the man is cleaning there"]),
        (["it has started to rain and a woman is opening an umbrella", "dark clouds are coming and someone has an umbrella"],
         ["it is doing something up there", "she has that thing open"]),
        (["a bus is coming around the corner", "the bus is pulling up to the bus stop"],
         ["something big is coming", "there is a thing coming there"]),
    ],
}

OPENERS = ["well", "okay", "alright", "let's see", "so", "in this picture", "I can see that", ""]
CONNECTORS = ["and", "and then", "also", "over here", "and I see", "then", "", ""]
FILLERS = ["uh", "um", "er", "uh uh", "um well", "hmm"]
UNCERTAIN = ["I don't know", "I can't remember what it's called", "what do you call it",
             "I'm not sure", "I can't see what else", "I forget", "I don't know what that is"]
CLOSERS_CONTROL = ["that's about everything I can see", "I think that covers it", "that's all", ""]
CLOSERS_DEMENTIA = ["that's all I know", "I don't know what else", "is that right", "that's it I guess", ""]


def make_transcript(rng: random.Random, label: str) -> str:
    impairment = rng.betavariate(1.6, 7.0) if label == "Control" else rng.betavariate(3.6, 3.2)
    facts = list(SCENES[rng.choice(list(SCENES))])
    if rng.random() < 0.3 + 0.5 * impairment:      # impaired speakers wander from the usual order
        rng.shuffle(facts)

    parts = []
    opener = rng.choice(OPENERS)
    if opener:
        parts.append(opener)
    for specific, vague in facts:
        if rng.random() > 0.95 - 0.55 * impairment:          # detail left out
            continue
        sentence = rng.choice(vague if rng.random() < 0.85 * impairment else specific)
        words = sentence.split()
        if rng.random() < 0.45 * impairment:                 # false start / repetition
            words = words[:rng.choice([1, 2])] + words
        if rng.random() < 0.06 + 0.55 * impairment:          # filler
            words.insert(rng.choice([0, 0, min(2, len(words))]), rng.choice(FILLERS))
        connector = rng.choice(CONNECTORS)
        parts.append((connector + " " if connector else "") + " ".join(words))
        if rng.random() < 0.30 * impairment:
            parts.append(rng.choice(UNCERTAIN))
        if rng.random() < 0.18 * impairment and parts:       # says the same thing again
            parts.append(parts[-1 if rng.random() < 0.5 else max(0, len(parts) - 2)])
    if not parts or len(" ".join(parts).split()) < 8:
        parts.append(rng.choice(UNCERTAIN))
        parts.append("there is something there " + rng.choice(FILLERS) + " I can't tell")
    closer = rng.choice(CLOSERS_CONTROL if impairment < 0.35 else CLOSERS_DEMENTIA)
    if closer:
        parts.append(closer)
    text = ". ".join(p[0].upper() + p[1:] for p in parts if p) + "."
    return text


def main(n_per_class: int = 250, seed: int = config.RANDOM_SEED) -> Path:
    rng = random.Random(seed)
    rows = []
    for label in config.SPEECH_CLASSES:
        for _ in range(n_per_class):
            rows.append([make_transcript(rng, label), label])
    rng.shuffle(rows)
    out = config.SYNTHETIC_SPEECH_CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text", "label", "source"])
        for i, (text, label) in enumerate(rows, 1):
            writer.writerow([f"SYN{i:04d}", text, label, "synthetic"])
    print(f"Wrote {len(rows)} synthetic transcripts to {out}")
    return out


if __name__ == "__main__":
    main()
