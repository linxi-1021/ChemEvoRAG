import type {
  ContentBlock,
  EvidenceBlock,
  MoleculeCard,
  ReactionEvent,
  MolNameLLM,
} from '../types';
import { getUuidFilename } from './manifest';

const BASE = '';  // Vite serves from root, data/ is at project root

async function fetchJson<T>(url: string): Promise<T> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Failed to fetch ${url}: ${resp.status}`);
  return resp.json();
}

async function fetchText(url: string): Promise<string> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Failed to fetch ${url}: ${resp.status}`);
  return resp.text();
}

/**
 * Load content_list_v2.json for a document.
 * Pages are the top-level array; each page is an array of ContentBlock.
 */
export async function loadContentListV2(docId: string): Promise<ContentBlock[][]> {
  const filename = getUuidFilename(docId);
  if (!filename) throw new Error(`No content_list_v2.json found for doc ${docId}`);
  return fetchJson<ContentBlock[][]>(`${BASE}/data/mineru_output/${docId}/extracted/${filename}`);
}

/** Load full.md markdown content */
export async function loadFullMarkdown(docId: string): Promise<string> {
  return fetchText(`${BASE}/data/mineru_output/${docId}/extracted/full.md`);
}

/** Load evidence blocks */
export async function loadEvidenceBlocks(docId: string): Promise<EvidenceBlock[]> {
  return fetchJson<EvidenceBlock[]>(`${BASE}/data/evidence/${docId}.blocks.json`);
}

/** Load evidence molecules */
export async function loadEvidenceMolecules(docId: string): Promise<MoleculeCard[]> {
  return fetchJson<MoleculeCard[]>(`${BASE}/data/evidence/${docId}.molecules.json`);
}

/** Load evidence reactions */
export async function loadEvidenceReactions(docId: string): Promise<ReactionEvent[]> {
  return fetchJson<ReactionEvent[]>(`${BASE}/data/evidence/${docId}.reactions.json`);
}

/** Load LLM molecule names */
export async function loadMolNames(docId: string): Promise<MolNameLLM[]> {
  return fetchJson<MolNameLLM[]>(`${BASE}/data/evidence/${docId}.mol_names_llm.json`);
}

/** Get URL for an extracted image */
export function getImageUrl(docId: string, imageId: string): string {
  return `${BASE}/data/mineru_output/${docId}/extracted/images/${imageId}.jpg`;
}

/** Get URL for the original PDF */
export function getPdfUrl(docId: string): string {
  return `${BASE}/data/pdfs/${docId}.pdf`;
}

// Layout.json types
export interface LayoutBlock {
  type: string;
  bbox: [number, number, number, number]; // PDF point coordinates
  angle?: number;
  content?: string | null;
  merge_prev?: boolean;
  text?: string;
  score?: number;
}

export interface LayoutPage {
  preproc_blocks: LayoutBlock[];
  discarded_blocks: LayoutBlock[];
  page_size: [number, number];
  page_idx: number;
  para_blocks?: unknown[];
}

export interface LayoutData {
  pdf_info: LayoutPage[];
  _backend?: string;
  _version_name?: string;
}

/** Load layout.json — bboxes are in PDF point coordinates (directly usable for iframe overlay) */
export async function loadLayout(docId: string): Promise<LayoutData> {
  return fetchJson<LayoutData>(`${BASE}/data/mineru_output/${docId}/extracted/layout.json`);
}
