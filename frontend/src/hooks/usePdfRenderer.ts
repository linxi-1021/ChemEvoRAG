import { useState, useEffect, useRef, useCallback } from 'react';
import * as pdfjsLib from 'pdfjs-dist';
import { useAppStore } from '../stores/useAppStore';
import pageDimensionsMap from '../api/page-dimensions.json';

pdfjsLib.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjsLib.version}/build/pdf.worker.min.mjs`

let pdfDocPromise: Promise<pdfjsLib.PDFDocumentProxy> | null = null;
let pdfDocInstance: pdfjsLib.PDFDocumentProxy | null = null;

/**
 * Manages PDF document loading. Returns a promise that resolves when PDF is ready.
 */
export function usePdfRenderer(pdfUrl: string | null) {
  const [totalPages, setTotalPages] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const setCurrentPage = useAppStore(s => s.setCurrentPage);
  const setTotalPagesStore = useAppStore(s => s.setTotalPages);

  useEffect(() => {
    if (!pdfUrl) return;

    let cancelled = false;
    setIsLoading(true);

    pdfDocPromise = pdfjsLib.getDocument(pdfUrl).promise.then(doc => {
      if (!cancelled) {
        pdfDocInstance = doc;
        setTotalPages(doc.numPages);
        setTotalPagesStore(doc.numPages);
        setIsLoading(false);
      }
      return doc;
    }).catch(err => {
      console.error('Failed to load PDF:', err);
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
      const doc = pdfDocInstance || (pdfDocPromise ? await pdfDocPromise : null);
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

  return { totalPages, isLoading, renderPage };
}
