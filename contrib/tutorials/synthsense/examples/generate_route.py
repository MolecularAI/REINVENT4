"""
Generate AIZynthFinder route JSON files with specified reaction classes.
"""

import json
import argparse
import sys


def build_route(reaction_classes: list) -> dict:
    """
    Build route with specified reaction classes.
    
    Uses a nested linear structure where each reaction's classification
    is set to the provided class code. All other data is template-based.
    """
    num_reactions = len(reaction_classes)
    
    # Build nested reactions recursively
    def build_reactions(classes):
        if not classes:
            return None
        
        reaction = {
            "children": [
                # First reactant (in stock)
                {
                    "hide": False,
                    "in_stock": True,
                    "is_chemical": True,
                    "smiles": "N",
                    "type": "mol",
                    "availableInReactionConnect": True
                },
                # Second reactant (intermediate or in stock)
                {
                    "hide": False,
                    "in_stock": len(classes) == 1,  # Only in stock if last reaction
                    "is_chemical": True,
                    "smiles": "O=Cc1cc(=O)[nH]o1",
                    "type": "mol",
                    "availableInReactionConnect": len(classes) == 1
                }
            ],
            "hide": False,
            "is_reaction": True,
            "metadata": {
                "classification": f"{classes[0]} Generated Reaction",
                "library_occurence": 750,
                "mapped_reaction_smiles": "[C:1]>>[C:1]",
                "policy_name": "reaction_connect_expansion",
                "policy_probability": 0.005,
                "policy_probability_rank": 1,
                "template": "[C:1]>>[C:1]",
                "template_code": 10000,
                "template_hash": "placeholder_hash"
            },
            "smiles": "[C:1]>>[C:1]",
            "type": "reaction"
        }
        
        # Add nested reaction if there are more classes
        if len(classes) > 1:
            reaction["children"][1]["children"] = [build_reactions(classes[1:])]
        
        return reaction
    
    # Root structure (target molecule)
    return {
        "children": [build_reactions(reaction_classes)],
        "hide": False,
        "in_stock": False,
        "is_chemical": True,
        "scores": {
            "average template occurrence": 780,
            "number of pre-cursors": num_reactions + 1,
            "number of pre-cursors in stock": num_reactions + 1,
            "number of reactions": num_reactions,
            "route cost": 971,
            "state score": 0.994039853898894,
            "sum of prices": 675
        },
        "smiles": "NCc1cc(=O)[nH]o1",
        "type": "mol",
        "availableInReactionConnect": False
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate AIZynthFinder route JSON files with specified reaction classes.')
    
    parser.add_argument(
        '-r', '--reactions',
        nargs='+',
        required=True,
        help='Reaction class codes (e.g., 3.1.1 2.22.32)'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output JSON file path (default: auto-generated)'
    )
    
    args = parser.parse_args()
    
    # Validate reaction class format (X.Y.Z where all are digits)
    for rc in args.reactions:
        parts = rc.split('.')
        if len(parts) < 2 or not all(p.isdigit() for p in parts):
            print(f"Error: Invalid format '{rc}'. Expected: X.Y.Z (e.g., 3.1.2)", file=sys.stderr)
            sys.exit(1)
    
    # Generate output filename if not provided
    if args.output is None:
        class_parts = [rc.replace('.', '') for rc in args.reactions]
        args.output = f"route_{'_'.join(class_parts)}.json"
    
    # Generate and save route
    route = build_route(args.reactions)
    
    with open(args.output, 'w') as f:
        json.dump(route, f, indent=4)
    
    print(f"Generated {len(args.reactions)}-step route: {args.output}")
    print(f"Reaction classes: {', '.join(args.reactions)}")


if __name__ == '__main__':
    main()
