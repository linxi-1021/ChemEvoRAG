import { create } from 'zustand';
import type { TabType } from '../types';

interface AppState {
  selectedDocId: string | null;
  currentPage: number;
  totalPages: number;
  scale: number;
  activeTab: TabType;

  // Cross-panel linking — highlightedBlockIds is a SET of evidence block IDs
  highlightedBlockIds: Set<string>;
  highlightedMoleculeId: string | null;
  highlightedReactionId: string | null;
  highlightedRectId: string | null; // for image/table persistent highlight
  scrollTargetId: string | null;
  // For image/table: scroll to matching markdown card by type + page + bbox position
  scrollToCardType: { type: string; pageNum: number; bbox?: [number, number, number, number] } | null;
  // For image/table: scroll PDF to page and highlight matching bbox
  scrollToPdfType: { type: string; pageNum: number; bbox?: [number, number, number, number] } | null;

  selectDocument: (docId: string) => void;
  setCurrentPage: (page: number) => void;
  setTotalPages: (n: number) => void;
  setScale: (s: number) => void;
  setActiveTab: (tab: TabType) => void;
  highlightBlocks: (blockIds: string[]) => void;
  highlightMolecule: (moleculeId: string | null) => void;
  highlightReaction: (reactionId: string | null) => void;
  setScrollTarget: (id: string | null) => void;
  scrollToCard: (type: string, pageNum: number, bbox?: [number, number, number, number]) => void;
  scrollToPdf: (type: string, pageNum: number, bbox?: [number, number, number, number]) => void;
  highlightRect: (rectId: string) => void;
  clearHighlight: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  selectedDocId: null,
  currentPage: 0,
  totalPages: 0,
  scale: 1.5,
  activeTab: 'markdown',
  highlightedBlockIds: new Set(),
  highlightedMoleculeId: null,
  highlightedReactionId: null,
  highlightedRectId: null,
  scrollTargetId: null,
  scrollToCardType: null,
  scrollToPdfType: null,

  selectDocument: (docId) => set({
    selectedDocId: docId,
    currentPage: 0,
    highlightedBlockIds: new Set(),
    highlightedMoleculeId: null,
    highlightedReactionId: null,
    scrollTargetId: null,
  }),

  setCurrentPage: (page) => set({ currentPage: page }),
  setTotalPages: (n) => set({ totalPages: n }),
  setScale: (s) => set({ scale: Math.max(0.5, Math.min(3, s)) }),
  setActiveTab: (tab) => set({ activeTab: tab }),

  // Accept multiple block IDs — a paragraph may span multiple bboxes
  highlightBlocks: (blockIds) => set({
    highlightedBlockIds: new Set(blockIds),
    highlightedMoleculeId: null,
    highlightedReactionId: null,
    highlightedRectId: null,
    scrollTargetId: blockIds[0] || null,
  }),

  highlightMolecule: (moleculeId) => set({
    highlightedMoleculeId: moleculeId,
    highlightedBlockIds: new Set(),
    highlightedReactionId: null,
    scrollTargetId: moleculeId,
  }),

  highlightReaction: (reactionId) => set({
    highlightedReactionId: reactionId,
    highlightedBlockIds: new Set(),
    highlightedMoleculeId: null,
    scrollTargetId: reactionId,
  }),

  setScrollTarget: (id) => set({ scrollTargetId: id }),

  scrollToCard: (type, pageNum, bbox) => set({
    scrollToCardType: { type, pageNum, bbox },
    highlightedBlockIds: new Set(),
    highlightedMoleculeId: null,
    highlightedReactionId: null,
  }),

  scrollToPdf: (type, pageNum, bbox) => set({
    scrollToPdfType: { type, pageNum, bbox },
    highlightedBlockIds: new Set(),
    highlightedMoleculeId: null,
    highlightedReactionId: null,
  }),

  highlightRect: (rectId) => set({
    highlightedRectId: rectId,
    highlightedBlockIds: new Set(),
    highlightedMoleculeId: null,
    highlightedReactionId: null,
  }),

  clearHighlight: () => set({
    highlightedBlockIds: new Set(),
    highlightedMoleculeId: null,
    highlightedReactionId: null,
    highlightedRectId: null,
    scrollTargetId: null,
    scrollToCardType: null,
    scrollToPdfType: null,
  }),
}));
