import { useMemo, useState, useRef, useCallback } from 'react';
import { useAppStore } from '../../stores/useAppStore';
import type { LayoutBlock, EvidenceBlock, MoleculeCard, ReactionEvent } from '../../types';
import styles from './BboxOverlay.module.css';

interface Props {
  layoutBlocks: LayoutBlock[];
  evidenceBlocks: EvidenceBlock[];
  molecules: MoleculeCard[];
  reactions: ReactionEvent[];
  scale: number;
  viewportWidth: number;
  viewportHeight: number;
  docId: string;
}

interface OverlayRect {
  id: string;
  paragraphBlockIds: string[];
  bbox: [number, number, number, number];
  label: string;
  layer: 'block' | 'molecule' | 'reaction';
}

/** Extract text from layout.json block (nested lines/spans) */
function extractLayoutText(block: LayoutBlock): string {
  if (block.text) return block.text;
  if (typeof block.content === 'string' && block.content) return block.content;
  if ('lines' in block && Array.isArray((block as any).lines)) {
    let text = '';
    for (const line of (block as any).lines) {
      for (const span of (line.spans || [])) {
        text += span.content || '';
      }
    }
    return text;
  }
  return '';
}

/**
 * Match layout blocks to evidence blocks via y-proximity grouping.
 *
 * 1. Skip image blocks and empty-text list blocks entirely.
 * 2. Group remaining blocks by y-center proximity (< 30 PDF-points = same paragraph).
 * 3. Each group consumes one evidence block sequentially.
 */
function matchBlocks(
  layoutBlocks: LayoutBlock[],
  evidenceBlocks: EvidenceBlock[],
): Map<number, string[]> {
  const result = new Map<number, string[]>();

  // Step 1: Filter to actionable blocks
  const actionable: { idx: number; block: LayoutBlock }[] = [];
  for (let i = 0; i < layoutBlocks.length; i++) {
    const lb = layoutBlocks[i];
    const text = extractLayoutText(lb);
    // Skip image blocks entirely
    if (lb.type === 'image') continue;
    // Skip empty-text list blocks (they don't carry evidence)
    if (lb.type === 'list' && !text) continue;
    actionable.push({ idx: i, block: lb });
  }

  // Step 2: Group by y-center proximity
  const Y_THRESHOLD = 30;
  const groups: { idx: number; block: LayoutBlock }[][] = [];

  for (const item of actionable) {
    const lastGroup = groups[groups.length - 1];
    if (lastGroup && lastGroup.length > 0) {
      const lastBlock = lastGroup[lastGroup.length - 1].block;
      const prevCenterY = (lastBlock.bbox[1] + lastBlock.bbox[3]) / 2;
      const curCenterY = (item.block.bbox[1] + item.block.bbox[3]) / 2;
      if (Math.abs(curCenterY - prevCenterY) < Y_THRESHOLD) {
        lastGroup.push(item);
        continue;
      }
    }
    groups.push([item]);
  }

  // Step 3: Assign each group the next evidence block
  let evIdx = 0;
  for (const group of groups) {
    if (evIdx >= evidenceBlocks.length) break;
    const evBlock = evidenceBlocks[evIdx];
    for (const g of group) {
      result.set(g.idx, [evBlock.block_id]);
    }
    evIdx++;
  }

  return result;
}

export default function BboxOverlay({
  layoutBlocks, evidenceBlocks, molecules, reactions,
  scale, viewportWidth, viewportHeight,
}: Props) {
  const highlightedBlockIds = useAppStore(s => s.highlightedBlockIds);
  const highlightedMoleculeId = useAppStore(s => s.highlightedMoleculeId);
  const highlightedReactionId = useAppStore(s => s.highlightedReactionId);
  const highlightBlocks = useAppStore(s => s.highlightBlocks);
  const highlightMolecule = useAppStore(s => s.highlightMolecule);
  const highlightReaction = useAppStore(s => s.highlightReaction);
  const setActiveTab = useAppStore(s => s.setActiveTab);

  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const dragRef = useRef<{ startX: number; startY: number; startOffX: number; startOffY: number } | null>(null);

  const blockMapping = useMemo(() => matchBlocks(layoutBlocks, evidenceBlocks), [layoutBlocks, evidenceBlocks]);

  const rects = useMemo(() => {
    const result: OverlayRect[] = [];
    for (let i = 0; i < layoutBlocks.length; i++) {
      const lb = layoutBlocks[i];
      if (!lb.bbox || lb.bbox[2] <= lb.bbox[0] || lb.bbox[3] <= lb.bbox[1]) continue;
      const text = extractLayoutText(lb);
      const groupIds = blockMapping.get(i) || [];
      result.push({
        id: `layout_${lb.type}_${lb.bbox[0]}_${lb.bbox[1]}`,
        paragraphBlockIds: groupIds,
        bbox: lb.bbox,
        label: `[${lb.type}] ${(text || '(cont)').slice(0, 60)}`,
        layer: 'block',
      });
    }
    for (const mol of molecules) {
      for (const si of mol.source_images) {
        if (si.provenance.bbox && si.provenance.bbox[2] > si.provenance.bbox[0]) {
          result.push({
            id: mol.molecule_card_id, paragraphBlockIds: [],
            bbox: si.provenance.bbox as [number, number, number, number],
            label: `Molecule: ${mol.canonical_smiles || '—'}`, layer: 'molecule',
          });
          break;
        }
      }
    }
    for (const rxn of reactions) {
      if (rxn.source?.bbox && rxn.source.bbox[2] > rxn.source.bbox[0]) {
        result.push({
          id: rxn.reaction_event_id, paragraphBlockIds: [],
          bbox: rxn.source.bbox as [number, number, number, number],
          label: `Reaction`, layer: 'reaction',
        });
      }
    }
    return result;
  }, [layoutBlocks, molecules, reactions, blockMapping]);

  const transformed = useMemo(() => {
    return rects.map(r => ({
      ...r,
      x: r.bbox[0] * scale + offset.x,
      y: r.bbox[1] * scale + offset.y,
      w: (r.bbox[2] - r.bbox[0]) * scale,
      h: (r.bbox[3] - r.bbox[1]) * scale,
    }));
  }, [rects, scale, offset]);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 1) return;
    e.preventDefault();
    dragRef.current = { startX: e.clientX, startY: e.clientY, startOffX: offset.x, startOffY: offset.y };
    const handleMouseMove = (me: MouseEvent) => {
      if (!dragRef.current) return;
      setOffset({
        x: dragRef.current.startOffX + me.clientX - dragRef.current.startX,
        y: dragRef.current.startOffY + me.clientY - dragRef.current.startY,
      });
    };
    const handleMouseUp = () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      if (dragRef.current) {
        const finalX = dragRef.current.startOffX + (event as MouseEvent).clientX - dragRef.current.startX;
        const finalY = dragRef.current.startOffY + (event as MouseEvent).clientY - dragRef.current.startY;
        console.log(`[BboxOverlay] Drag offset: x=${finalX.toFixed(1)}, y=${finalY.toFixed(1)}`);
        console.log(`[BboxOverlay] Apply to usePdfRenderer: offsetX=${finalX.toFixed(1)}, offsetY=${finalY.toFixed(1)}`);
      }
      dragRef.current = null;
    };
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }, [offset]);

  const handleClick = (rect: OverlayRect) => {
    if (rect.layer === 'block' && rect.paragraphBlockIds.length > 0) {
      highlightBlocks(rect.paragraphBlockIds);
      setActiveTab('markdown');
    } else if (rect.layer === 'molecule') {
      highlightMolecule(rect.id);
    } else if (rect.layer === 'reaction') {
      highlightReaction(rect.id);
    }
  };

  const getHighlighted = (rect: OverlayRect): boolean => {
    if (rect.layer === 'block') return rect.paragraphBlockIds.some(id => highlightedBlockIds.has(id));
    if (rect.layer === 'molecule') return highlightedMoleculeId === rect.id;
    if (rect.layer === 'reaction') return highlightedReactionId === rect.id;
    return false;
  };

  return (
    <svg
      className={styles.overlay}
      width={viewportWidth}
      height={viewportHeight}
      viewBox={`0 0 ${viewportWidth} ${viewportHeight}`}
      onMouseDown={handleMouseDown}
      style={{ cursor: 'grab' }}
    >
      {transformed.map(rect => {
        if (rect.w <= 0 || rect.h <= 0) return null;
        const isHighlighted = getHighlighted(rect);
        const layerClass = styles[rect.layer] || '';
        return (
          <rect
            key={`${rect.layer}-${rect.id}`}
            x={rect.x} y={rect.y} width={rect.w} height={rect.h}
            className={`${styles.rect} ${layerClass} ${isHighlighted ? styles.highlighted : ''}`}
            onClick={() => handleClick(rect)}
          >
            <title>{rect.label}</title>
          </rect>
        );
      })}
    </svg>
  );
}
