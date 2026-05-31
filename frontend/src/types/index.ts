// Content List V2 types (from MinerU output)
export interface ContentTextItem {
  type: string;
  content: string;
}

export interface TitleContent {
  title_content: ContentTextItem[];
  level: number;
}

export interface ParagraphContent {
  paragraph_content: ContentTextItem[];
}

export interface ImageSource {
  path: string;
}

export interface ImageContent {
  image_source: ImageSource;
  content: string;
  image_caption: ContentTextItem[];
  image_footnote: ContentTextItem[];
}

export interface ListItem {
  item_type: string;
  item_content: ContentTextItem[];
}

export interface ListContent {
  list_type: string;
  list_items: ListItem[];
}

export interface TableContent {
  image_source?: ImageSource;
  table_caption: ContentTextItem[];
  table_footnote: ContentTextItem[];
  html?: string;
  table_type?: string;
  table_nest_level?: number;
}

export interface ContentBlock {
  type: 'title' | 'paragraph' | 'image' | 'list' | 'table' | 'page_header' | 'page_footer' | 'page_number' | 'page_aside_text' | string;
  bbox: [number, number, number, number];
  content: TitleContent | ParagraphContent | ImageContent | ListContent | TableContent;
  level?: number;
  sub_type?: string;
}

// Evidence types
export interface EvidenceBlock {
  block_id: string;
  doc_id: string;
  block_type: string;
  text: string;
  page: number;
  bbox: [number, number, number, number];
  section: string;
  prev_block_id: string | null;
  next_block_id: string | null;
  source_file: string;
  confidence: number;
  raw_payload: Record<string, unknown>;
  errors: string[];
}

export interface SourceProvenance {
  doc_id: string;
  source_file: string | null;
  page: number;
  bbox: [number, number, number, number] | null;
  block_id: string | null;
  figure_id: string | null;
  table_id: string | null;
  section: string | null;
  text_span: string | null;
}

export interface SourceImage {
  image_id: string;
  figure_id: string | null;
  provenance: SourceProvenance;
  bbox: [number, number, number, number];
  confidence: number;
}

export interface MoleculeCard {
  molecule_card_id: string;
  doc_id: string;
  local_ids: string[];
  names: string[];
  raw_smiles: string | null;
  canonical_smiles: string | null;
  inchi_key: string | null;
  iupac_name: string | null;
  aliases: string[];
  source_mentions: string[];
  source_images: SourceImage[];
  linked_elementkg_id: string | null;
  confidence: number;
  normalization_status: string;
  raw_payload: Record<string, unknown>;
  errors: string[];
}

export interface ReactionComponent {
  name: string;
  smiles: string;
  role: 'reactant' | 'product' | 'reagent' | 'catalyst' | 'solvent' | 'unknown';
  linked_molecule_card_id: string | null;
  amount: string | null;
  confidence: number | null;
}

export interface YieldValue {
  value: number;
  unit: string;
  normalized_value: number;
  raw_text: string;
}

export interface ReactionEvent {
  reaction_event_id: string;
  doc_id: string;
  reaction_smiles: string | null;
  reactants: ReactionComponent[];
  products: ReactionComponent[];
  reagents: ReactionComponent[];
  catalysts: ReactionComponent[];
  solvents: ReactionComponent[];
  temperature: string | null;
  time: string | null;
  yield_value: YieldValue | null;
  procedure_text: string | null;
  source: SourceProvenance;
  supporting_block_ids: string[];
  supporting_table_ids: string[];
  supporting_figure_ids: string[];
  confidence: number;
  evidence_completeness: number | null;
  raw_payload: Record<string, unknown>;
  errors: string[];
}

export interface MolNameLLM {
  compound_label: string;
  name: string;
  aliases: string[];
  page: number;
  evidence: string;
}

// App state types
export interface DocumentMeta {
  docId: string;
  label: string;
}

export type TabType = 'markdown' | 'chemical' | 'json';

// Markdown parsed block
export interface MarkdownBlock {
  id: string;
  type: 'heading' | 'paragraph' | 'image' | 'table' | 'chemical' | 'footnote';
  content: string;
  level?: number;
  imageUrl?: string;
  imageId?: string;
  evidenceBlockId?: string;
  pageIndex?: number;
}
