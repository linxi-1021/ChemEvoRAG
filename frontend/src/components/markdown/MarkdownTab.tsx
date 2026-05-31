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
  const scrollTargetId = useAppStore(s => s.scrollTargetId);
  const highlightBlocks = useAppStore(s => s.highlightBlocks);
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
    }[] = [];

    let blockIdx = 0;

    for (let pageIdx = 0; pageIdx < docData.contentList.length; pageIdx++) {
      const page = docData.contentList[pageIdx];
      const pageNum = pageIdx + 1;

      for (const clBlock of page) {
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

  const handleBlockClick = (block: typeof structuredBlocks[0]) => {
    if (block.evidenceBlockId) {
      // Find ALL markdown blocks that share the same evidence block ID
      // (same paragraph group — a paragraph may span multiple content_list_v2 blocks)
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
