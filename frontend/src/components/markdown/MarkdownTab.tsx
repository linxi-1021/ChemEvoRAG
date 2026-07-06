import { useMemo, useRef, useEffect } from 'react';
import { useAppStore } from '../../stores/useAppStore';
import { getImageUrl } from '../../api/dataLoader';
import type { DocumentData } from '../../hooks/useDocumentData';
import type { ContentBlock } from '../../types';
import type { EvidenceBlock } from '../../types';
import styles from './MarkdownTab.module.css';

interface Props {
  docId: string;
  docData: DocumentData;
}

/** Extract text from content_list_v2 block content */
function extractText(content: ContentBlock['content']): string {
  if ('paragraph_content' in content) {
    return content.paragraph_content.map(c => c.content).join('');
  }
  if ('title_content' in content) {
    return content.title_content.map(c => c.content).join('');
  }
  if ('list_items' in content) {
    return content.list_items.map(item =>
      item.item_content.map(c => c.content).join('')
    ).join('\n');
  }
  return '';
}

/** Get image path from content_list_v2 image block */
function getImagePath(content: ContentBlock['content']): string | null {
  if ('image_source' in content) {
    return content.image_source.path;
  }
  return null;
}

/** Get table HTML from content_list_v2 table block */
function getTableHtml(content: ContentBlock['content']): string | null {
  if ('html' in content) {
    return (content as { html?: string }).html || null;
  }
  return null;
}

/** Match a content_list_v2 block to an evidence block by text similarity */
function matchEvidenceBlock(
  clBlock: ContentBlock,
  evidenceBlocks: EvidenceBlock[],
  page: number,
): EvidenceBlock | null {
  const clText = extractText(clBlock.content).trim().slice(0, 80).toLowerCase();
  if (!clText) return null;

  let bestMatch: EvidenceBlock | null = null;
  let bestScore = 0;

  for (const eb of evidenceBlocks) {
    if (eb.page !== page) continue;
    const ebText = eb.text.trim().slice(0, 80).toLowerCase();
    let overlap = 0;
    const minLen = Math.min(clText.length, ebText.length);
    for (let i = 0; i < minLen; i++) {
      if (clText[i] === ebText[i]) overlap++;
      else break;
    }
    if (overlap > bestScore && overlap >= Math.min(15, clText.length)) {
      bestScore = overlap;
      bestMatch = eb;
    }
  }

  return bestMatch;
}

export default function MarkdownTab({ docId, docData }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const highlightedBlockIds = useAppStore(s => s.highlightedBlockIds);
  const highlightedRectId = useAppStore(s => s.highlightedRectId);
  const scrollTargetId = useAppStore(s => s.scrollTargetId);
  const scrollToCardType = useAppStore(s => s.scrollToCardType);
  const scrollToPdfType = useAppStore(s => s.scrollToPdfType);
  const highlightBlocks = useAppStore(s => s.highlightBlocks);
  const scrollToPdf = useAppStore(s => s.scrollToPdf);
  const setCurrentPage = useAppStore(s => s.setCurrentPage);

  // Build structured blocks from content_list_v2 + evidence
  const structuredBlocks = useMemo(() => {
    if (!docData.contentList) return [];

    const blocks: {
      id: string;
      type: string;
      text: string;
      page: number;
      evidenceBlockId: string | null;
      imageUrl: string | null;
      tableHtml: string | null;
      level?: number;
      bbox?: [number, number, number, number];
    }[] = [];

    let blockIdx = 0;

    for (let pageIdx = 0; pageIdx < docData.contentList.length; pageIdx++) {
      const page = docData.contentList[pageIdx];
      const pageNum = pageIdx + 1;

      for (const clBlock of page) {
        // Skip page headers, footers, and page numbers (not content)
        if (clBlock.type === 'page_header' || clBlock.type === 'page_footer' || clBlock.type === 'page_number') continue;

        const text = extractText(clBlock.content);
        if (!text && clBlock.type !== 'image' && clBlock.type !== 'table') continue;

        const evidenceMatch = matchEvidenceBlock(clBlock, docData.evidenceBlocks, pageNum);
        const imageUrl = clBlock.type === 'image' ? getImagePath(clBlock.content) : null;
        const tableHtml = clBlock.type === 'table' ? getTableHtml(clBlock.content) : null;
        const level = 'level' in clBlock ? clBlock.level : undefined;

        blocks.push({
          id: `cl_${blockIdx++}`,
          type: clBlock.type,
          text,
          page: pageNum,
          evidenceBlockId: evidenceMatch?.block_id || null,
          imageUrl: imageUrl ? getImageUrl(docId, imageUrl.replace('images/', '').replace('.jpg', '')) : null,
          tableHtml,
          level,
          bbox: clBlock.bbox as [number, number, number, number] | undefined,
        });
      }
    }

    return blocks;
  }, [docData.contentList, docData.evidenceBlocks, docId]);

  // Auto-scroll to highlighted block
  useEffect(() => {
    if (!scrollTargetId || !containerRef.current) return;
    const el = containerRef.current.querySelector(`[data-block-id="${scrollTargetId}"]`);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [scrollTargetId]);

  // Scroll to matching markdown card when image/table is clicked in PDF
  useEffect(() => {
    if (!scrollToCardType || !containerRef.current) return;
    const { type, pageNum: pg, bbox: srcBbox } = scrollToCardType;

    // Find candidate cards by type and page
    const cards = containerRef.current.querySelectorAll('[data-block-id]');
    const candidates: { card: Element; bbox: [number, number, number, number] }[] = [];
    for (const card of Array.from(cards)) {
      const typeEl = card.querySelector('[class*="typeLabel"]');
      const pageEl = card.querySelector('[class*="pageBadge"]');
      if (typeEl && typeEl.textContent === type && pageEl && pageEl.textContent === `P${pg}`) {
        // Get bbox from structuredBlocks data attribute or fallback
        const bboxStr = card.getAttribute('data-bbox');
        if (bboxStr) {
          candidates.push({ card, bbox: JSON.parse(bboxStr) });
        } else {
          candidates.push({ card, bbox: [0, 0, 0, 0] });
        }
      }
    }

    if (candidates.length === 0) return;

    // Find closest by y-center if source bbox is provided
    let best = candidates[0];
    if (srcBbox) {
      const srcYCenter = (srcBbox[1] + srcBbox[3]) / 2;
      let bestDist = Infinity;
      for (const c of candidates) {
        const cy = (c.bbox[1] + c.bbox[3]) / 2;
        const dist = Math.abs(cy - srcYCenter);
        if (dist < bestDist) {
          bestDist = dist;
          best = c;
        }
      }
    }

    best.card.scrollIntoView({ behavior: 'smooth', block: 'center' });
    // Persistent highlight — stays until highlightedRectId changes
    best.card.style.borderColor = 'var(--color-primary)';
    best.card.style.background = 'rgba(22, 119, 255, 0.08)';
    best.card.style.boxShadow = '0 0 0 2px rgba(22, 119, 255, 0.15)';
  }, [scrollToCardType]);

  // Highlight matching markdown card when image/table card is clicked in markdown
  useEffect(() => {
    if (!scrollToPdfType || !containerRef.current) return;
    const { type, pageNum: pg, bbox: srcBbox } = scrollToPdfType;
    const cards = containerRef.current.querySelectorAll('[data-block-id]');
    const candidates: { card: Element; bbox: [number, number, number, number] }[] = [];
    for (const card of Array.from(cards)) {
      const typeEl = card.querySelector('[class*="typeLabel"]');
      const pageEl = card.querySelector('[class*="pageBadge"]');
      if (typeEl && typeEl.textContent === type && pageEl && pageEl.textContent === `P${pg}`) {
        const bboxStr = card.getAttribute('data-bbox');
        candidates.push({ card, bbox: bboxStr ? JSON.parse(bboxStr) : [0, 0, 0, 0] });
      }
    }
    if (candidates.length === 0) return;
    let best = candidates[0];
    if (srcBbox) {
      const srcYCenter = (srcBbox[1] + srcBbox[3]) / 2;
      let bestDist = Infinity;
      for (const c of candidates) {
        const cy = (c.bbox[1] + c.bbox[3]) / 2;
        const dist = Math.abs(cy - srcYCenter);
        if (dist < bestDist) { bestDist = dist; best = c; }
      }
    }
    best.card.scrollIntoView({ behavior: 'smooth', block: 'center' });
    // Persistent highlight — stays until highlightedRectId changes
    best.card.style.borderColor = 'var(--color-primary)';
    best.card.style.background = 'rgba(22, 119, 255, 0.08)';
    best.card.style.boxShadow = '0 0 0 2px rgba(22, 119, 255, 0.15)';
  }, [scrollToPdfType]);

  // Clear card highlight when highlightedRectId changes to null (text block clicked)
  useEffect(() => {
    if (highlightedRectId || !containerRef.current) return;
    const cards = containerRef.current.querySelectorAll('[data-block-id]');
    for (const card of Array.from(cards)) {
      if (card.style.borderColor) {
        card.style.borderColor = '';
        card.style.background = '';
        card.style.boxShadow = '';
      }
    }
  }, [highlightedRectId]);

  const handleBlockClick = (block: typeof structuredBlocks[0]) => {
    if (block.type === 'image' || block.type === 'table') {
      // Image/table — scroll PDF to the page and highlight matching bbox
      scrollToPdf(block.type, block.page, block.bbox);
    } else if (block.evidenceBlockId) {
      // Text block — find ALL markdown blocks that share the same evidence block ID
      const allMatchingIds = structuredBlocks
        .filter(b => b.evidenceBlockId === block.evidenceBlockId)
        .map(b => b.evidenceBlockId!);
      highlightBlocks([...new Set(allMatchingIds)]);
      setCurrentPage(block.page - 1);
    }
  };

  if (!docData.contentList) {
    return <div className={styles.empty}>No content available.</div>;
  }

  return (
    <div ref={containerRef} className={styles.container}>
      {structuredBlocks.map(block => (
        <div
          key={block.id}
          data-block-id={block.evidenceBlockId || block.id}
          data-bbox={block.bbox ? JSON.stringify(block.bbox) : undefined}
          className={`${styles.card} ${
            block.evidenceBlockId && highlightedBlockIds.has(block.evidenceBlockId) ? styles.highlighted : ''
          }`}
          onClick={() => handleBlockClick(block)}
        >
          <div className={styles.header}>
            <span className={styles.typeIcon}>
              {block.type === 'title' ? '#' :
               block.type === 'image' ? '🖼' :
               block.type === 'table' ? '📊' : '¶'}
            </span>
            <span className={styles.typeLabel}>{block.type}</span>
            {block.level && <span className={styles.level}>H{block.level}</span>}
            {block.page && <span className={styles.pageBadge}>P{block.page}</span>}
          </div>

          <div className={styles.content}>
            {block.imageUrl ? (
              <div className={styles.imageWrapper}>
                <img src={block.imageUrl} alt="" className={styles.image} loading="lazy" />
              </div>
            ) : block.tableHtml ? (
              <div className={styles.tableContent} dangerouslySetInnerHTML={{ __html: block.tableHtml }} />
            ) : (
              <div className={styles.textContent}>
                {block.text.length > 400 ? block.text.slice(0, 400) + '...' : block.text}
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
