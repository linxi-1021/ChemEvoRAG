import { useMemo, useRef, useEffect, useState } from 'react';
import { useAppStore } from '../../stores/useAppStore';
import { getPdfUrl } from '../../api/dataLoader';
import { usePdfRenderer } from '../../hooks/usePdfRenderer';
import BboxOverlay from './BboxOverlay';
import type { DocumentData } from '../../hooks/useDocumentData';
import type { LayoutBlock } from '../../api/dataLoader';
import styles from './PdfViewer.module.css';

interface Props {
  docId: string;
  docData: DocumentData;
}

export default function PdfViewer({ docId, docData }: Props) {
  const pdfUrl = getPdfUrl(docId);
  const scrollRef = useRef<HTMLDivElement>(null);
  const currentPage = useAppStore(s => s.currentPage);
  const setCurrentPage = useAppStore(s => s.setCurrentPage);
  const totalPages = docData.contentList?.length || 0;
  const { renderPage } = usePdfRenderer(pdfUrl);

  const programmaticScrollRef = useRef(false);

  useEffect(() => {
    const container = scrollRef.current;
    if (!container) return;
    const handleScroll = () => {
      if (programmaticScrollRef.current) return;
      const containerRect = container.getBoundingClientRect();
      const containerCenter = containerRect.top + containerRect.height / 2;
      const pageElements = container.querySelectorAll('[data-page-num]');
      let closestPage = 1;
      let minDistance = Infinity;
      pageElements.forEach(el => {
        const pageNum = parseInt(el.getAttribute('data-page-num') || '1', 10);
        const rect = el.getBoundingClientRect();
        const pageCenter = rect.top + rect.height / 2;
        const distance = Math.abs(pageCenter - containerCenter);
        if (distance < minDistance) { minDistance = distance; closestPage = pageNum; }
      });
      setCurrentPage(closestPage - 1);
    };
    container.addEventListener('scroll', handleScroll, { passive: true });
    return () => container.removeEventListener('scroll', handleScroll);
  }, [setCurrentPage]);

  useEffect(() => {
    if (!scrollRef.current) return;
    programmaticScrollRef.current = true;
    const timer = setTimeout(() => {
      if (!scrollRef.current) return;
      const pageEl = scrollRef.current.querySelector(`[data-page-num="${currentPage + 1}"]`);
      if (pageEl) pageEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
      setTimeout(() => { programmaticScrollRef.current = false; }, 500);
    }, 50);
    return () => { clearTimeout(timer); programmaticScrollRef.current = false; };
  }, [currentPage]);

  const pageNumbers = useMemo(
    () => Array.from({ length: totalPages }, (_, i) => i + 1),
    [totalPages],
  );

  return (
    <div className={styles.pdfViewer}>
      <div className={styles.pageHeader}>
        <span className={styles.pageInfo}>
          {totalPages > 0 ? `Page ${currentPage + 1} / ${totalPages}` : 'Loading...'}
        </span>
      </div>
      <div ref={scrollRef} className={styles.scrollContainer}>
        {pageNumbers.map(pageNum => (
          <PdfPageCanvas
            key={pageNum}
            pageNum={pageNum}
            docId={docId}
            docData={docData}
            renderPage={renderPage}
          />
        ))}
      </div>
    </div>
  );
}

function PdfPageCanvas({
  pageNum, docId, docData, renderPage,
}: {
  pageNum: number; docId: string; docData: DocumentData;
  renderPage: (pageNum: number, canvas: HTMLCanvasElement, docId: string) => Promise<{ width: number; height: number } | null>;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [dims, setDims] = useState<{ width: number; height: number } | null>(null);
  const [renderScale, setRenderScale] = useState(2.0);
  const renderingRef = useRef(false);

  useEffect(() => {
    if (!canvasRef.current || renderingRef.current) return;
    renderingRef.current = true;
    let cancelled = false;
    renderPage(pageNum, canvasRef.current, docId).then(result => {
      if (!cancelled && result) {
        setDims({ width: result.width, height: result.height });
        setRenderScale(result.renderScale);
      }
      renderingRef.current = false;
    });
    return () => { cancelled = true; renderingRef.current = false; };
  }, [pageNum, renderPage, docId]);

  // Layout.json blocks (PDF point coordinates)
  const layoutBlocks: LayoutBlock[] = useMemo(() => {
    if (!docData.layout?.pdf_info) return [];
    const page = docData.layout.pdf_info[pageNum - 1];
    return page?.preproc_blocks || [];
  }, [docData.layout, pageNum]);

  // Evidence blocks for this page
  const pageEvidenceBlocks = useMemo(() => {
    return docData.evidenceBlocks.filter(b => b.page === pageNum);
  }, [docData.evidenceBlocks, pageNum]);

  const pageMolecules = useMemo(() => {
    return docData.molecules.filter(m =>
      m.source_images.some(si => si.provenance.page === pageNum)
    );
  }, [docData.molecules, pageNum]);

  const pageReactions = useMemo(() => {
    return docData.reactions.filter(r => r.source?.page === pageNum);
  }, [docData.reactions, pageNum]);

  // layout.json bbox × renderScale = canvas pixels (linear mapping)
  const bboxScale = renderScale;

  return (
    <div className={styles.pageWrapper} data-page-num={pageNum}>
      <div className={styles.canvasWrapper} style={dims ? { width: dims.width, height: dims.height } : undefined}>
        <canvas ref={canvasRef} className={styles.canvas} />
        {dims && (
          <BboxOverlay
            layoutBlocks={layoutBlocks}
            evidenceBlocks={pageEvidenceBlocks}
            molecules={pageMolecules}
            reactions={pageReactions}
            scale={bboxScale}
            viewportWidth={dims.width}
            viewportHeight={dims.height}
            docId={docId}
          />
        )}
      </div>
    </div>
  );
}
