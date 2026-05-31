import { useCallback } from 'react';
import { useAppStore } from '../../stores/useAppStore';
import { getImageUrl } from '../../api/dataLoader';
import styles from './MarkdownBlockCard.module.css';
import type { MarkdownBlock } from '../../types';

interface Props {
  block: MarkdownBlock;
  docId: string;
  evidenceBlockId: string | null;
  pageIndex?: number;
  isCrossPage: boolean;
  pageRange: string;
}

export default function MarkdownBlockCard({
  block,
  docId,
  evidenceBlockId,
  pageIndex,
  isCrossPage,
  pageRange,
}: Props) {
  const highlightedBlockId = useAppStore(s => s.highlightedBlockId);
  const highlightBlock = useAppStore(s => s.highlightBlock);
  const setCurrentPage = useAppStore(s => s.setCurrentPage);
  const clearHighlight = useAppStore(s => s.clearHighlight);

  const isHighlighted = highlightedBlockId === evidenceBlockId;

  const handleClick = useCallback(() => {
    if (evidenceBlockId) {
      highlightBlock(evidenceBlockId);
      if (pageIndex !== undefined) {
        setCurrentPage(pageIndex);
      }
    }
  }, [evidenceBlockId, pageIndex, highlightBlock, setCurrentPage]);

  const handleCopy = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    navigator.clipboard.writeText(block.content);
  }, [block.content]);

  const handleMouseLeave = useCallback(() => {
    // Delayed clear so rapid re-entering doesn't flicker
    setTimeout(() => {
      const store = useAppStore.getState();
      if (store.highlightedBlockId === evidenceBlockId) {
        clearHighlight();
      }
    }, 300);
  }, [evidenceBlockId, clearHighlight]);

  const typeIcon: Record<string, string> = {
    heading: '#',
    paragraph: '¶',
    image: '🖼',
    table: '📊',
    chemical: '⚗',
    footnote: '¹',
  };

  const imageUrl = block.imageId ? getImageUrl(docId, block.imageId) : null;

  return (
    <div
      data-block-id={evidenceBlockId || block.id}
      className={`${styles.card} ${isHighlighted ? styles.highlighted : ''}`}
      onClick={handleClick}
      onMouseLeave={handleMouseLeave}
    >
      <div className={styles.header}>
        <span className={styles.typeIcon}>{typeIcon[block.type] || '•'}</span>
        <span className={styles.typeLabel}>{block.type}</span>
        {block.level && <span className={styles.level}>H{block.level}</span>}
        {isCrossPage && (
          <span className={styles.crossPageBadge}>跨页 {pageRange}</span>
        )}
        {pageIndex !== undefined && (
          <span className={styles.pageBadge}>P{pageIndex + 1}</span>
        )}
        <button className={styles.copyBtn} onClick={handleCopy} title="Copy">
          📋
        </button>
      </div>

      <div className={styles.content}>
        {block.type === 'image' && imageUrl ? (
          <div className={styles.imageWrapper}>
            <img
              src={imageUrl}
              alt="Extracted figure"
              className={styles.image}
              loading="lazy"
            />
          </div>
        ) : block.type === 'table' ? (
          <div
            className={styles.tableContent}
            dangerouslySetInnerHTML={{ __html: block.content }}
          />
        ) : (
          <div className={styles.textContent}>
            {block.content.length > 300
              ? block.content.slice(0, 300) + '...'
              : block.content}
          </div>
        )}
      </div>
    </div>
  );
}
