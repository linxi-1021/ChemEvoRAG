import type { DocumentMeta } from '../types';
import uuidManifest from './uuid-manifest.json';

export const DOC_IDS = [
  '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
  '11', '12', '13', '14', '15-SI', '16', '17', '18',
  '19', '20', '21', '22', '23',
];

export const DOCUMENTS: DocumentMeta[] = DOC_IDS.map(id => ({
  docId: id,
  label: `Paper ${id}`,
}));

/**
 * Get the UUID-prefixed filename for content_list_v2.json.
 * Uses a pre-generated manifest mapping doc_id → filename.
 */
export function getUuidFilename(docId: string): string | null {
  return (uuidManifest as Record<string, string>)[docId] || null;
}
