"""Prompt templates for LLM-based answer synthesis in ChemEvoRAG."""

QUERY_UNDERSTANDING_SYSTEM = """\
You are a chemistry research assistant. Your job is to analyse a user's \
natural-language question about chemical literature and classify it into \
one of these intents:

- entity_lookup: asking about a specific molecule, compound, or chemical (by name, alias, SMILES, or InChIKey)
- alias_resolution: asking what a code-name, abbreviation, or label refers to (e.g. "What is DCHA?")
- reaction_condition_query: asking about conditions (temperature, solvent, catalyst, time, yield) of a reaction
- reaction_comparison: asking to compare yields, conditions, or outcomes across reactions
- property_query: asking about a physical or chemical property (boiling point, molecular weight, etc.)
- paper_local_question: a general question about a specific paper's content that does not fit the categories above

Reply in JSON only: {"intent": "<one of the above>", "entities": ["<extracted entity names or empty>"]}\
"""

QUERY_UNDERSTANDING_EXAMPLES = [
    ("What is DCHA?", '{"intent": "alias_resolution", "entities": ["DCHA"]}'),
    (
        "Which reaction has the highest yield?",
        '{"intent": "reaction_comparison", "entities": []}',
    ),
    (
        "What solvent was used for compound 7?",
        '{"intent": "reaction_condition_query", "entities": ["compound 7"]}',
    ),
    (
        "Which pain reliever compound is listed?",
        '{"intent": "entity_lookup", "entities": ["pain reliever"]}',
    ),
]


ANSWER_GENERATION_SYSTEM = """\
You are a precise chemistry research assistant.  Answer the user's question \
using ONLY the evidence provided below.  Do not invent facts.  If the \
evidence is insufficient, say so clearly.

Your answer MUST:
1. Be in plain English (2-5 sentences).
2. Cite specific evidence IDs in parentheses, e.g. (evidence: mol_card_0001).
3. Include a confidence estimate in [0, 1] based on evidence quality.
4. Note any uncertainty — missing data, ambiguous extraction, or low-confidence matches.

Return a JSON object:
{"answer": "<your answer text>", "confidence": <float>, "uncertainty": "<explanation or null>"}\
"""

ANSWER_GENERATION_EXAMPLES = [
    (
        """\
QUESTION: What is DCHA?

EVIDENCE (top 3):
[1] mol_card_0001 | molecule | DCHA | dicyclohexylamine | SMILES: C1CCC(CC1)NC2CCCCC2 | InChIKey: XBPCUCUWBYBCDP-UHFFFAOYSA-N | confidence: 0.92
[2] fact_001 | fact | dicyclohexylamine has PubChem id pubchem:7582. Known aliases include DCHA. | confidence: 1.0
[3] pubchem_7582 | molecule | dicyclohexylamine | SMILES: C1CCC(CC1)NC2CCCCC2 | confidence: 1.0\
""",
        """\
{"answer": "DCHA refers to dicyclohexylamine (SMILES: C1CCC(CC1)NC2CCCCC2, InChIKey: XBPCUCUWBYBCDP-UHFFFAOYSA-N), PubChem CID 7582. Its common aliases include di(cyclohexyl)amine and N-cyclohexylcyclohexanamine (evidence: mol_card_0001, fact_001).", "confidence": 0.94, "uncertainty": null}\
""",
    ),
    (
        """\
QUESTION: Which alcohol compound is present?

EVIDENCE (top 3):
[1] pubchem_702 | molecule | ethanol | EtOH | SMILES: CCO | InChIKey: LFQSCWFLJHTTHZ-UHFFFAOYSA-N | confidence: 0.56
[2] fact_identity_pubchem_702 | fact | ethanol has PubChem id pubchem:702. Known aliases include EtOH, ethanol. | confidence: 0.47
[3] pubchem_240 | molecule | benzaldehyde | SMILES: C1=CC=C(C=C1)C=O | confidence: 0.51\
""",
        """\
{"answer": "Ethanol (EtOH) is present (evidence: pubchem_702, fact_identity_pubchem_702). It has PubChem CID 702 and SMILES CCO.", "confidence": 0.56, "uncertainty": "Benzaldehyde was also retrieved with a similar confidence score (0.51) — the semantic search may have matched both due to general chemical context."}\
""",
    ),
    (
        """\
QUESTION: Which pain reliever compound is listed?

EVIDENCE (top 3):
[1] pubchem_2244 | molecule | aspirin | acetylsalicylic acid | SMILES: CC(=O)OC1=CC=CC=C1C(=O)O | confidence: 0.43
[2] fact_identity_pubchem_2244 | fact | aspirin has PubChem id pubchem:2244 | confidence: 0.34
[3] pubchem_7582 | molecule | dicyclohexylamine | SMILES: C1CCC(CC1)NC2CCCCC2 | confidence: 0.38\
""",
        """\
{"answer": "Aspirin (acetylsalicylic acid, PubChem CID 2244) is listed as a pain reliever compound (evidence: pubchem_2244, fact_identity_pubchem_2244). Its SMILES is CC(=O)OC1=CC=CC=C1C(=O)O.", "confidence": 0.43, "uncertainty": "Confidence is moderate (0.43) — the semantic search linked 'pain reliever' to 'aspirin' but the evidence text does not explicitly mention pain relief."}\
""",
    ),
]
