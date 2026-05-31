import { useState, useEffect } from 'react';
import { useAppStore } from '../stores/useAppStore';
import {
  loadContentListV2,
  loadFullMarkdown,
  loadEvidenceBlocks,
  loadEvidenceMolecules,
  loadEvidenceReactions,
  loadMolNames,
  loadLayout,
} from '../api/dataLoader';
import type {
  ContentBlock,
  EvidenceBlock,
  MoleculeCard,
  ReactionEvent,
  MolNameLLM,
} from '../types';
import type { LayoutData } from '../api/dataLoader';

export interface DocumentData {
  contentList: ContentBlock[][] | null;
  fullMarkdown: string | null;
  evidenceBlocks: EvidenceBlock[];
  molecules: MoleculeCard[];
  reactions: ReactionEvent[];
  molNames: MolNameLLM[];
  layout: LayoutData | null;
  isLoading: boolean;
  error: string | null;
}

export function useDocumentData(): DocumentData {
  const selectedDocId = useAppStore(s => s.selectedDocId);
  const [data, setData] = useState<DocumentData>({
    contentList: null,
    fullMarkdown: null,
    evidenceBlocks: [],
    molecules: [],
    reactions: [],
    molNames: [],
    layout: null,
    isLoading: false,
    error: null,
  });

  useEffect(() => {
    if (!selectedDocId) {
      setData({
        contentList: null,
        fullMarkdown: null,
        evidenceBlocks: [],
        molecules: [],
        reactions: [],
        molNames: [],
        layout: null,
        isLoading: false,
        error: null,
      });
      return;
    }

    let cancelled = false;
    setData(prev => ({ ...prev, isLoading: true, error: null }));

    async function load() {
      try {
        const [contentList, fullMarkdown, evidenceBlocks, molecules, reactions, molNames, layout] =
          await Promise.allSettled([
            loadContentListV2(selectedDocId!),
            loadFullMarkdown(selectedDocId!),
            loadEvidenceBlocks(selectedDocId!),
            loadEvidenceMolecules(selectedDocId!),
            loadEvidenceReactions(selectedDocId!),
            loadMolNames(selectedDocId!),
            loadLayout(selectedDocId!),
          ]);

        if (cancelled) return;

        setData({
          contentList: contentList.status === 'fulfilled' ? contentList.value : null,
          fullMarkdown: fullMarkdown.status === 'fulfilled' ? fullMarkdown.value : null,
          evidenceBlocks: evidenceBlocks.status === 'fulfilled' ? evidenceBlocks.value : [],
          molecules: molecules.status === 'fulfilled' ? molecules.value : [],
          reactions: reactions.status === 'fulfilled' ? reactions.value : [],
          molNames: molNames.status === 'fulfilled' ? molNames.value : [],
          layout: layout.status === 'fulfilled' ? layout.value : null,
          isLoading: false,
          error: null,
        });
      } catch (err) {
        if (!cancelled) {
          setData(prev => ({
            ...prev,
            isLoading: false,
            error: err instanceof Error ? err.message : 'Unknown error',
          }));
        }
      }
    }

    load();
    return () => { cancelled = true; };
  }, [selectedDocId]);

  return data;
}
