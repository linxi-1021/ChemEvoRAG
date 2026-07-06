import { useState, useEffect, useRef, useCallback } from 'react';
import * as pdfjsLib from 'pdfjs-dist';
import { useAppStore } from '../stores/useAppStore';
import pageDimensionsMap from '../api/page-dimensions.json';

pdfjsLib.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjsLib.version}/build/pdf.worker.min.mjs`

// Module-level: one promise per URL, shared across renders
let currentUrl = '';
let currentDocPromise: Promise<pdfjsLib.PDFDocumentProxy> | null = null;
let currentDoc: pdfjsLib.PDFDocumentProxy | null = null;

/**
 * Manages PDF document loading.
 */
export function usePdfRenderer(pdfUrl: string | null) {
  const [totalPages, setTotalPages] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [docGeneration, setDocGeneration] = useState(0);
  const setCurrentPage = useAppStore(s => s.setCurrentPage);
  const setTotalPagesStore = useAppStore(s => s.setTotalPages);

  useEffect(() => {
    if (!pdfUrl) return;
    // Already loaded this URL
    if (pdfUrl === currentUrl && currentDoc) {
      setTotalPages(currentDoc.numPages);
      setTotalPagesStore(currentDoc.numPages);
      return;
    }

    let cancelled = false;
    setIsLoading(true);
    setDocGeneration(g => g + 1);

    // Destroy old document
    if (currentDoc) {
      currentDoc.destroy();
      currentDoc = null;
    }

    currentUrl = pdfUrl;
    currentDocPromise = pdfjsLib.getDocument(pdfUrl).promise.then(doc => {
      if (cancelled) {
        doc.destroy();
        throw new Error('cancelled');
      }
      currentDoc = doc;
      setTotalPages(doc.numPages);
      setTotalPagesStore(doc.numPages);
      setIsLoading(false);
      return doc;
    }).catch(err => {
      if (!cancelled) console.error('Failed to load PDF:', err);
      throw err;
    });

    return () => { cancelled = true; };
  }, [pdfUrl]);

  /** Render a single page. Scale computed to match MinerU canvas width. */
  const renderPage = useCallback(async (
    pageNum: number,
    canvas: HTMLCanvasElement,
    docId: string,
  ): Promise<{ width: number; height: number; renderScale: number } | null> => {
    try {
      // Wait for document to be ready
      const doc = currentDoc || (currentDocPromise ? await currentDocPromise.catch(() => null) : null);
      if (!doc) return null;

      const page = await doc.getPage(pageNum);
      const mineruW = (pageDimensionsMap as Record<string, number[][]>)?.[docId]?.[pageNum - 1]?.[0] || 1000;
      const pdfPageW = page.getViewport({ scale: 1 }).width;
      const renderScale = mineruW / pdfPageW;

      const viewport = page.getViewport({ scale: renderScale });
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      const ctx = canvas.getContext('2d')!;
      await page.render({ canvasContext: ctx, viewport }).promise;

      return { width: viewport.width, height: viewport.height, renderScale };
    } catch (err) {
      if (err instanceof Error && err.message?.includes('cancelled')) return null;
      console.error(`Failed to render page ${pageNum}:`, err);
      return null;
    }
  }, []);

  return { totalPages, isLoading, renderPage, docGeneration };
}
