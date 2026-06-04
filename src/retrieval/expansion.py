"""Graph-based evidence expansion via internal evidence links.

Expands candidate evidence by following internal links:
  - MoleculeCard source_mentions → related DocumentBlocks
  - ReactionEventCard supporting_block_ids → related DocumentBlocks
  - DocumentBlock prev/next → neighboring blocks
  - DocumentBlock section → same-section blocks
"""

from __future__ import annotations

from evidence import (
    CandidateEvidence,
    DocumentBlock,
    MoleculeCard,
    ReactionEventCard,
    SourceProvenance,
)
from storage import LocalStore


def expand_evidence(
    candidates: list,
    store: LocalStore,
    top_n: int = 5,
) -> list:
    """基于内链扩展证据。

    Args:
        candidates: 当前候选证据列表
        store: 本地 evidence 存储
        top_n: 对前 N 个候选做扩展

    Returns:
        扩展后的候选列表（包含原始候选 + 新扩展的证据）
    """
    expanded = list(candidates)
    seen_ids: set[str] = set(c.evidence_id for c in candidates)

    for c in candidates[:top_n]:
        if c.evidence_type == "molecule":
            expanded_cards = _expand_from_molecule(c, store, seen_ids)
            expanded.extend(expanded_cards)
            for ec in expanded_cards:
                seen_ids.add(ec.evidence_id)

        elif c.evidence_type == "reaction_event":
            expanded_cards = _expand_from_reaction(c, store, seen_ids)
            expanded.extend(expanded_cards)
            for ec in expanded_cards:
                seen_ids.add(ec.evidence_id)

        elif c.evidence_type == "document_block":
            expanded_cards = _expand_from_block(c, store, seen_ids)
            expanded.extend(expanded_cards)
            for ec in expanded_cards:
                seen_ids.add(ec.evidence_id)

    return expanded


def _expand_from_molecule(
    c: CandidateEvidence, store: LocalStore, seen_ids: set[str]
) -> list:
    """从分子候选中扩展关联的 block。"""
    result = []
    slots = c.structured_slots or {}
    doc_id = slots.get("doc_id", "")

    # 通过 source_mentions 的 block_id 找到关联的 DocumentBlock
    # 从 MoleculeCard 的 source_mentions 中提取 block_id
    mol_card_id = _extract_mol_card_id(c.evidence_id)
    if not mol_card_id or not doc_id:
        return result

    try:
        mols = store.load_molecules(doc_id)
    except Exception:
        return result

    for mol in mols:
        if mol.molecule_card_id == mol_card_id:
            for mention in mol.source_mentions:
                if mention.provenance and mention.provenance.block_id:
                    block_id = mention.provenance.block_id
                    if block_id not in seen_ids:
                        block = _load_block(store, doc_id, block_id)
                        if block:
                            result.append(_block_to_candidate(block, mention.provenance))
            break

    return result


def _expand_from_reaction(
    c: CandidateEvidence, store: LocalStore, seen_ids: set[str]
) -> list:
    """从反应候选中扩展关联的 block。"""
    result = []
    slots = c.structured_slots or {}
    doc_id = slots.get("doc_id", "")

    if not doc_id:
        return result

    # 通过 supporting_block_ids 扩展
    for block_id in slots.get("supporting_blocks", []) or []:
        if block_id not in seen_ids:
            block = _load_block(store, doc_id, block_id)
            if block:
                src = SourceProvenance(
                    doc_id=doc_id,
                    source_file=block.source_file,
                    page=block.page,
                    bbox=block.bbox,
                    block_id=block.block_id,
                    section=block.section,
                )
                result.append(_block_to_candidate(block, src))

    return result


def _expand_from_block(
    c: CandidateEvidence, store: LocalStore, seen_ids: set[str]
) -> list:
    """从 block 候选中扩展相邻上下文。"""
    result = []
    slots = c.structured_slots or {}
    doc_id = slots.get("doc_id", "")
    block_id = c.evidence_id

    if not doc_id:
        return result

    try:
        blocks = store.load_blocks(doc_id)
    except Exception:
        return result

    for block in blocks:
        if block.block_id == block_id:
            # 扩展前一个 block
            if block.prev_block_id and block.prev_block_id not in seen_ids:
                prev = _find_block(blocks, block.prev_block_id)
                if prev:
                    src = SourceProvenance(
                        doc_id=doc_id,
                        source_file=prev.source_file,
                        page=prev.page,
                        bbox=prev.bbox,
                        block_id=prev.block_id,
                        section=prev.section,
                    )
                    result.append(_block_to_candidate(prev, src))

            # 扩展后一个 block
            if block.next_block_id and block.next_block_id not in seen_ids:
                nxt = _find_block(blocks, block.next_block_id)
                if nxt:
                    src = SourceProvenance(
                        doc_id=doc_id,
                        source_file=nxt.source_file,
                        page=nxt.page,
                        bbox=nxt.bbox,
                        block_id=nxt.block_id,
                        section=nxt.section,
                    )
                    result.append(_block_to_candidate(nxt, src))
            break

    return result


def _extract_mol_card_id(evidence_id: str) -> str | None:
    """从 evidence_id 提取 MoleculeCard 的 ID。

    匹配模式: mol_*, mol_tbl_*, mol_txt_*
    """
    if evidence_id.startswith("mol_"):
        return evidence_id
    return None


def _load_block(store: LocalStore, doc_id: str, block_id: str) -> DocumentBlock | None:
    """加载特定的 DocumentBlock。"""
    try:
        blocks = store.load_blocks(doc_id)
    except Exception:
        return None
    return _find_block(blocks, block_id)


def _find_block(blocks: list, block_id: str) -> DocumentBlock | None:
    """在 block 列表中查找指定 block_id。"""
    for b in blocks:
        if b.block_id == block_id:
            return b
    return None


def _block_to_candidate(block: DocumentBlock, source: SourceProvenance) -> CandidateEvidence:
    """将 DocumentBlock 转为 CandidateEvidence。"""
    return CandidateEvidence(
        evidence_id=block.block_id,
        evidence_type="document_block",
        summary=block.text,
        structured_slots={
            "doc_id": block.doc_id,
            "block_type": block.block_type,
            "score": 0.5,
        },
        source=source,
        confidence=0.5,
    )
