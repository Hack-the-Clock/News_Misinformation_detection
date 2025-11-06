"""
Generate mock LLM extraction data for GNN training

This creates realistic entity/event/relation extractions without making actual LLM calls.
Use this for training when API limits are reached. For production demo, use real LLM calls.
"""

import json
import random
from datetime import datetime
from pathlib import Path
from src.data.fever_loader import FEVERDatasetLoader

# Mock extraction templates based on FEVER claim patterns
def generate_mock_extraction(claim_text, label):
    """
    Generate a realistic mock extraction based on the claim text

    Args:
        claim_text: The FEVER claim text
        label: 'SUPPORTS' or 'REFUTES'

    Returns:
        Mock extraction dict with entities, events, and relations
    """

    # Extract potential entities from the claim using simple heuristics
    words = claim_text.split()

    # Mock entity types
    entities = []
    events = []
    relations = []

    # Pattern 1: Simple factual claims (e.g., "X is a Y")
    if " is a " in claim_text or " is an " in claim_text:
        parts = claim_text.replace(" is a ", "|").replace(" is an ", "|").split("|")
        if len(parts) >= 2:
            subject = parts[0].strip().rstrip('.')
            object_type = parts[1].strip().rstrip('.')

            entities.append({
                "name": subject,
                "type": "person" if subject[0].isupper() else "concept",
                "properties": {}
            })

            # Add profession/type as event
            events.append({
                "description": f"{subject} is a {object_type}",
                "temporal_expression": "",
                "entities_involved": [subject],
                "properties": {}
            })

    # Pattern 2: Location claims (e.g., "X is set in Y", "X is located in Y")
    elif " is set in " in claim_text or " is located in " in claim_text or " is in " in claim_text:
        for pattern in [" is set in ", " is located in ", " is in "]:
            if pattern in claim_text:
                parts = claim_text.replace(pattern, "|").split("|")
                if len(parts) >= 2:
                    subject = parts[0].strip()
                    location = parts[1].strip().rstrip('.')

                    entities.append({
                        "name": subject,
                        "type": "concept",
                        "properties": {}
                    })

                    entities.append({
                        "name": location,
                        "type": "location",
                        "properties": {}
                    })

                    relations.append({
                        "source": subject,
                        "relation": "set_in" if "set in" in pattern else "located_in",
                        "target": location,
                        "temporal_expression": ""
                    })
                break

    # Pattern 3: Action/work claims (e.g., "X worked with Y", "X starred in Y")
    elif " worked with " in claim_text or " starred in " in claim_text or " appeared in " in claim_text:
        for pattern, relation_type in [(" worked with ", "worked_with"), (" starred in ", "starred_in"), (" appeared in ", "appeared_in")]:
            if pattern in claim_text:
                parts = claim_text.replace(pattern, "|").split("|")
                if len(parts) >= 2:
                    subject = parts[0].strip()
                    object_entity = parts[1].strip().rstrip('.')

                    entities.append({
                        "name": subject,
                        "type": "person",
                        "properties": {}
                    })

                    entities.append({
                        "name": object_entity,
                        "type": "organization" if "Company" in object_entity or "Corporation" in object_entity else "concept",
                        "properties": {}
                    })

                    relations.append({
                        "source": subject,
                        "relation": relation_type,
                        "target": object_entity,
                        "temporal_expression": ""
                    })
                break

    # Pattern 4: Generic - extract capitalized words as entities
    else:
        # Find capitalized sequences (likely proper nouns)
        capitalized_sequences = []
        current_seq = []
        for word in words:
            clean_word = word.strip('.,!?;:')
            if clean_word and clean_word[0].isupper():
                current_seq.append(clean_word)
            else:
                if current_seq:
                    capitalized_sequences.append(" ".join(current_seq))
                    current_seq = []
        if current_seq:
            capitalized_sequences.append(" ".join(current_seq))

        # Add unique capitalized sequences as entities
        for seq in capitalized_sequences[:3]:  # Limit to 3 entities
            entity_type = "person" if len(seq.split()) <= 2 else "organization"
            entities.append({
                "name": seq,
                "type": entity_type,
                "properties": {}
            })

        # Create a simple event from the claim
        if entities:
            events.append({
                "description": claim_text.rstrip('.'),
                "temporal_expression": "",
                "entities_involved": [entities[0]["name"]],
                "properties": {}
            })

    # If we didn't extract much, create fallback entities
    if len(entities) < 1:
        # Extract first capitalized word as entity
        for word in words:
            clean_word = word.strip('.,!?;:')
            if clean_word and clean_word[0].isupper():
                entities.append({
                    "name": clean_word,
                    "type": "person",
                    "properties": {}
                })
                break

    # Add some randomness for REFUTES claims (contradictory info)
    if label == 'REFUTES' and random.random() > 0.5:
        # Add a contradictory relation
        if len(entities) >= 2:
            relations.append({
                "source": entities[0]["name"],
                "relation": "contradicts",
                "target": entities[1]["name"] if len(entities) > 1 else "statement",
                "temporal_expression": ""
            })

    return {
        "entities": entities,
        "events": events,
        "relations": relations
    }


def create_mock_llm_log(num_samples=100):
    """
    Create mock LLM call log with extractions from FEVER dataset

    Args:
        num_samples: Number of samples to generate
    """
    print("="*80)
    print("MOCK LLM DATA GENERATOR")
    print("="*80)
    print()
    print(f"Generating {num_samples} mock LLM extractions from FEVER dataset...")
    print()

    # Load FEVER dataset
    loader = FEVERDatasetLoader()
    balanced_data = loader.get_balanced_dataset(n_samples=num_samples, split='train')

    print(f"✓ Loaded {len(balanced_data)} FEVER examples")
    print()

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    log_file = "logs/llm_calls.log"

    # Generate mock extractions
    print(f"📝 Generating mock extractions...")

    with open(log_file, 'w') as f:
        for i, example in enumerate(balanced_data):
            # Generate mock extraction
            extraction = generate_mock_extraction(
                example['text'],
                example['original_label']
            )

            # Create log entry in same format as real LLM calls
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "method": "extract_structured_data",
                "payload": {
                    "text": example['text'],
                    "system_prompt": "Mock extraction (generated for training)",
                    "response_format": "json",
                    "temperature": 0.1
                },
                "response": extraction
            }

            f.write(json.dumps(log_entry) + "\n")

            if (i + 1) % 20 == 0:
                print(f"  Generated {i+1}/{len(balanced_data)} extractions...")

    print()
    print(f"✅ Successfully created {len(balanced_data)} mock LLM extractions")
    print(f"   Saved to: {log_file}")
    print()

    # Show statistics
    labels_count = {"SUPPORTS": 0, "REFUTES": 0}
    total_entities = 0
    total_events = 0
    total_relations = 0

    with open(log_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            response = entry['response']
            total_entities += len(response.get('entities', []))
            total_events += len(response.get('events', []))
            total_relations += len(response.get('relations', []))

    for ex in balanced_data:
        labels_count[ex['original_label']] = labels_count.get(ex['original_label'], 0) + 1

    print("📊 Mock Data Statistics:")
    print(f"   Total extractions: {len(balanced_data)}")
    print(f"   SUPPORTS (label=0): {labels_count.get('SUPPORTS', 0)}")
    print(f"   REFUTES (label=1): {labels_count.get('REFUTES', 0)}")
    print(f"   Avg entities per extraction: {total_entities/len(balanced_data):.2f}")
    print(f"   Avg events per extraction: {total_events/len(balanced_data):.2f}")
    print(f"   Avg relations per extraction: {total_relations/len(balanced_data):.2f}")
    print()

    print("💡 Next Steps:")
    print("   1. Run: python train_gnn.py --use_llm_log --llm_log_limit 100 --epochs 20")
    print("   2. This will train the GNN on the mock data")
    print("   3. For demo, use real LLM calls: python demo.py")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate mock LLM extraction data')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of mock samples to generate (default: 100)')

    args = parser.parse_args()

    create_mock_llm_log(num_samples=args.num_samples)
