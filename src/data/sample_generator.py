"""Curated benchmark sample dataset generator for offline reproducibility and testing."""

import json
from pathlib import Path
from typing import List, Dict, Any, Union
from src.data.schema import ClaimEvidenceItem
from src.data.loader import split_dataset, save_split_to_jsonl
from src.utils.logger import setup_logger

logger = setup_logger("sample_generator")

# Representative curated benchmark samples reflecting LLM hallucination modes in HaluEval format
CURATED_BENCHMARK_SAMPLES: List[Dict[str, Any]] = [
    {
        "knowledge": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889.",
        "question": "Where is the Eiffel Tower located and who built it?",
        "right_answer": "The Eiffel Tower is located on the Champ de Mars in Paris, France, and was built by Gustave Eiffel's company.",
        "hallucinated_answer": "The Eiffel Tower is located in Rome, Italy, and was built by the Roman Emperor Nero in 1889.",
    },
    {
        "knowledge": "Albert Einstein received the 1921 Nobel Prize in Physics for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect.",
        "question": "Why did Albert Einstein win the Nobel Prize in Physics?",
        "right_answer": "Einstein received the 1921 Nobel Prize in Physics for his discovery of the law of the photoelectric effect.",
        "hallucinated_answer": "Albert Einstein won the Nobel Prize in Physics for inventing the atomic bomb and developing quantum teleportation.",
    },
    {
        "knowledge": "Photosynthesis is a biological process used by plants, algae, and certain bacteria to convert light energy into chemical energy stored in glucose, releasing oxygen as a byproduct.",
        "question": "What gas is released as a byproduct during photosynthesis?",
        "right_answer": "Oxygen is released as a byproduct of photosynthesis while glucose stores the chemical energy.",
        "hallucinated_answer": "Photosynthesis releases large amounts of pure nitrogen gas and methane as direct byproducts.",
    },
    {
        "knowledge": "The James Webb Space Telescope (JWST) is an infrared space observatory launched on December 25, 2021, operating around the Sun-Earth L2 Lagrange point.",
        "question": "When was the James Webb Space Telescope launched and what spectrum does it observe?",
        "right_answer": "The James Webb Space Telescope was launched on December 25, 2021, and observes primarily in the infrared spectrum.",
        "hallucinated_answer": "The James Webb Space Telescope was launched in 1990 to observe exclusively ultraviolet rays from low Earth orbit.",
    },
    {
        "knowledge": "DNA (deoxyribonucleic acid) is composed of two polynucleotide chains that coil around each other to form a double helix carrying genetic instructions, using four bases: adenine (A), cytosine (C), guanine (G), and thymine (T).",
        "question": "What are the four nucleotide bases found in DNA?",
        "right_answer": "DNA contains four nucleotide bases: adenine, cytosine, guanine, and thymine arranged in a double helix.",
        "hallucinated_answer": "DNA is composed of five nucleotide bases including adenine, uracil, alanine, lysine, and glucose.",
    },
    {
        "knowledge": "The Pacific Ocean is the largest and deepest of Earth's five oceanic divisions. It extends from the Arctic Ocean in the north to the Southern Ocean in the south.",
        "question": "What is the largest ocean on Earth?",
        "right_answer": "The Pacific Ocean is the largest and deepest ocean on Earth, spanning from the Arctic to the Southern Ocean.",
        "hallucinated_answer": "The Atlantic Ocean is the largest and deepest ocean on Earth, surpassing all other oceanic divisions.",
    },
    {
        "knowledge": "The Python programming language was created by Guido van Rossum and was first released on February 20, 1991. Python emphasizes code readability with significant indentation.",
        "question": "Who created Python and when was it released?",
        "right_answer": "Python was created by Guido van Rossum and released in 1991, emphasizing clean syntax and readability.",
        "hallucinated_answer": "Python was created by Bjarne Stroustrup in 1972 at Bell Labs to replace C++.",
    },
    {
        "knowledge": "The Great Wall of China is a series of fortifications that were built across the historical northern borders of ancient Chinese states and Imperial China as protection against nomadic groups.",
        "question": "Why was the Great Wall of China built?",
        "right_answer": "The Great Wall of China was constructed to protect ancient Chinese states against nomadic invasions along northern borders.",
        "hallucinated_answer": "The Great Wall of China was built as a giant commercial aqueduct to transport freshwater across the Sahara desert.",
    },
    {
        "knowledge": "Penicillin was discovered in 1928 by Scottish scientist Alexander Fleming as an antibiotic substance produced by the Penicillium notatum mold.",
        "question": "Who discovered penicillin and in what year?",
        "right_answer": "Alexander Fleming discovered penicillin in 1928 as an antibacterial compound derived from mold.",
        "hallucinated_answer": "Penicillin was synthesized by Marie Curie in 1945 during her research on radioactivity.",
    },
    {
        "knowledge": "Jupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant with a mass more than two and a half times that of all the other planets combined.",
        "question": "What kind of planet is Jupiter and where is it located?",
        "right_answer": "Jupiter is a massive gas giant and is the fifth planet from the Sun.",
        "hallucinated_answer": "Jupiter is a small terrestrial rocky planet situated between Mercury and Venus.",
    },
    {
        "knowledge": "The human heart is a muscular organ with four chambers: the right atrium, right ventricle, left atrium, and left ventricle, responsible for pumping blood through the circulatory system.",
        "question": "How many chambers does the human heart have?",
        "right_answer": "The human heart consists of four chambers: two atria and two ventricles for circulatory pumping.",
        "hallucinated_answer": "The human heart possesses only two chambers that filter toxins directly from the bloodstream.",
    },
    {
        "knowledge": "Alan Turing was an English mathematician and computer scientist widely considered to be the father of theoretical computer science and artificial intelligence.",
        "question": "What is Alan Turing known for?",
        "right_answer": "Alan Turing is considered the father of theoretical computer science and artificial intelligence.",
        "hallucinated_answer": "Alan Turing was an 18th-century French general who commanded naval fleets in the American Civil War.",
    },
    {
        "knowledge": "Water has a chemical formula of H2O, meaning each molecule consists of one oxygen atom covalently bonded to two hydrogen atoms.",
        "question": "What is the molecular composition of water?",
        "right_answer": "A water molecule is composed of two hydrogen atoms bonded to one oxygen atom (H2O).",
        "hallucinated_answer": "Water is composed of two helium atoms bonded to one carbon atom (He2C).",
    },
    {
        "knowledge": "The Apollo 11 mission was the first spaceflight that landed humans on the Moon on July 20, 1969, with astronauts Neil Armstrong and Buzz Aldrin.",
        "question": "When did humans first land on the Moon and on which mission?",
        "right_answer": "Humans first landed on the Moon on July 20, 1969, during the Apollo 11 mission.",
        "hallucinated_answer": "Humans first landed on the Moon in 1984 during the Voyager 2 robotic probe mission.",
    },
    {
        "knowledge": "The speed of light in vacuum is exactly 299,792,458 meters per second, commonly denoted as c in physics equations.",
        "question": "What is the exact speed of light in vacuum?",
        "right_answer": "The speed of light in a vacuum is exactly 299,792,458 meters per second.",
        "hallucinated_answer": "The speed of light in a vacuum is approximately 343 meters per second, identical to the speed of sound in air.",
    },
    {
        "knowledge": "The Mona Lisa is a half-length portrait painting by Italian artist Leonardo da Vinci, created during the Italian Renaissance and displayed in the Louvre Museum in Paris.",
        "question": "Who painted the Mona Lisa and where is it displayed?",
        "right_answer": "The Mona Lisa was painted by Leonardo da Vinci and is permanently displayed in the Louvre Museum in Paris.",
        "hallucinated_answer": "The Mona Lisa was painted by Vincent van Gogh in Amsterdam and is exhibited in Tokyo.",
    },
    {
        "knowledge": "Transformers in deep learning are neural network architectures relying on self-attention mechanisms, introduced by Vaswani et al. in the 2017 paper 'Attention Is All You Need'.",
        "question": "Who introduced the Transformer architecture and in what year?",
        "right_answer": "The Transformer architecture was introduced in 2017 by Vaswani et al. using self-attention mechanisms.",
        "hallucinated_answer": "The Transformer architecture was invented by Yann LeCun in 1989 for convolutional digit recognition.",
    },
    {
        "knowledge": "The Sahara is the largest hot desert in the world and the third-largest desert overall, covering most of North Africa.",
        "question": "Where is the Sahara Desert located and how large is it?",
        "right_answer": "The Sahara is the largest hot desert in the world, spanning across northern Africa.",
        "hallucinated_answer": "The Sahara Desert is a frozen polar plateau situated in southern South America.",
    },
    {
        "knowledge": "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas with an elevation of 8,848.86 meters.",
        "question": "What is the highest mountain on Earth and what is its elevation?",
        "right_answer": "Mount Everest is the highest mountain on Earth, standing at 8,848.86 meters in the Himalayas.",
        "hallucinated_answer": "Mount Kilimanjaro is Earth's highest peak at an elevation of over 15,000 meters above sea level.",
    },
    {
        "knowledge": "The Industrial Revolution was the transition to new manufacturing processes in Great Britain, continental Europe, and the United States, that occurred during the period from around 1760 to 1840.",
        "question": "When and where did the Industrial Revolution begin?",
        "right_answer": "The Industrial Revolution began in Great Britain around 1760, transitioning to mechanized manufacturing.",
        "hallucinated_answer": "The Industrial Revolution began in ancient Greece around 500 BC through the use of digital electronics.",
    },
]


def generate_benchmark_dataset(
    output_raw_file: Union[str, Path] = "data/raw/halueval_sample.jsonl",
    output_split_dir: Union[str, Path] = "data/processed",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate and partition representative benchmark dataset for development and exploration.

    Args:
        output_raw_file: Path to write raw JSONL file.
        output_split_dir: Directory to store train.jsonl, val.jsonl, and test.jsonl.
        train_ratio: Training partition proportion.
        val_ratio: Validation partition proportion.
        test_ratio: Testing partition proportion.
        seed: Random seed.

    Returns:
        Summary dictionary with counts and paths.
    """
    raw_path = Path(output_raw_file)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    with open(raw_path, "w", encoding="utf-8") as f:
        for sample in CURATED_BENCHMARK_SAMPLES:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info(f"Generated raw benchmark dataset with {len(CURATED_BENCHMARK_SAMPLES)} samples at: {raw_path}")

    # Parse into ClaimEvidenceItems
    from src.data.loader import HaluEvalLoader
    loader = HaluEvalLoader()
    items = loader.load_from_file(raw_path)

    # Split into train, val, test
    split = split_dataset(
        items,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        stratify=True,
    )

    # Save to processed directory
    paths = save_split_to_jsonl(split, output_split_dir)

    return {
        "raw_file": str(raw_path),
        "raw_samples_count": len(CURATED_BENCHMARK_SAMPLES),
        "total_claim_items": len(items),
        "train_size": split.train_size,
        "val_size": split.val_size,
        "test_size": split.test_size,
        "split_paths": paths,
    }


def generate_synthetic_halueval_dataset(num_samples: int = 40) -> List[Dict[str, Any]]:
    """Return synthetic sample list balanced with factual (0) and hallucinated (1) items."""
    result = []
    idx = 0
    while len(result) < num_samples:
        for item in CURATED_BENCHMARK_SAMPLES:
            # Add factual sample
            result.append({
                "claim_id": f"c_{idx:03d}",
                "claim_text": item["right_answer"],
                "evidence_text": item["knowledge"],
                "binary_label": 0,
                "hallucination_label": 0,
                "right_answer": item["right_answer"],
                "hallucinated_answer": item["hallucinated_answer"],
                "knowledge_context": item["knowledge"],
            })
            idx += 1
            if len(result) >= num_samples:
                break

            # Add hallucinated sample
            result.append({
                "claim_id": f"c_{idx:03d}",
                "claim_text": item["hallucinated_answer"],
                "evidence_text": item["knowledge"],
                "binary_label": 1,
                "hallucination_label": 1,
                "right_answer": item["right_answer"],
                "hallucinated_answer": item["hallucinated_answer"],
                "knowledge_context": item["knowledge"],
            })
            idx += 1
            if len(result) >= num_samples:
                break
    return result
