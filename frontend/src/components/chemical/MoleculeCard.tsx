import { useRef, useEffect, useCallback } from 'react';
import { useAppStore } from '../../stores/useAppStore';
import { getImageUrl } from '../../api/dataLoader';
import type { MoleculeCard as MoleculeCardType, MolNameLLM } from '../../types';
import styles from './MoleculeCard.module.css';

interface Props {
  molecule: MoleculeCardType;
  docId: string;
  index: number;
  molName?: MolNameLLM;
}

export default function MoleculeCard({ molecule, docId, index, molName }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const highlightedMoleculeId = useAppStore(s => s.highlightedMoleculeId);
  const highlightMolecule = useAppStore(s => s.highlightMolecule);
  const setCurrentPage = useAppStore(s => s.setCurrentPage);

  const isHighlighted = highlightedMoleculeId === molecule.molecule_card_id;

  // Render SMILES as 2D structure
  useEffect(() => {
    const smiles = molecule.canonical_smiles || molecule.raw_smiles;
    if (!smiles || !canvasRef.current) return;

    // Check if smiles-drawer is available
    import('smiles-drawer').then(({ default: SmilesDrawer }) => {
      if (!canvasRef.current) return;
      try {
        const drawer = new SmilesDrawer.SmiDrawer({ width: 180, height: 140 });
        SmilesDrawer.parse(smiles, (tree: unknown) => {
          drawer.draw(tree as never, canvasRef.current!, 'light');
        }, () => {
          // Parse failed — draw fallback
          const ctx = canvasRef.current!.getContext('2d')!;
          ctx.fillStyle = '#f5f5f5';
          ctx.fillRect(0, 0, 180, 140);
          ctx.fillStyle = '#999';
          ctx.font = '11px sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText('Invalid SMILES', 90, 75);
        });
      } catch {
        // Fallback
      }
    });
  }, [molecule.canonical_smiles, molecule.raw_smiles]);

  const handleClick = useCallback(() => {
    if (molecule.source_images.length > 0) {
      const src = molecule.source_images[0];
      if (src.provenance.page) {
        setCurrentPage(src.provenance.page - 1);
      }
    }
    highlightMolecule(molecule.molecule_card_id);
  }, [molecule, setCurrentPage, highlightMolecule]);

  const sourceImage = molecule.source_images[0];
  const figureUrl = sourceImage
    ? getImageUrl(docId, sourceImage.image_id)
    : null;

  // Crop molecule from source figure using normalized bbox
  const croppedRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (!figureUrl || !sourceImage?.bbox || !croppedRef.current) return;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const canvas = croppedRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d')!;
      const [x1, y1, x2, y2] = sourceImage.bbox;
      const sx = x1 * img.naturalWidth;
      const sy = y1 * img.naturalHeight;
      const sw = (x2 - x1) * img.naturalWidth;
      const sh = (y2 - y1) * img.naturalHeight;
      canvas.width = Math.max(1, Math.round(sw));
      canvas.height = Math.max(1, Math.round(sh));
      ctx.drawImage(img, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
    };
    img.src = figureUrl;
  }, [figureUrl, sourceImage]);

  const smiles = molecule.canonical_smiles || molecule.raw_smiles || '—';
  const displayName = molName?.name || molecule.names?.[0] || molecule.aliases?.[0] || '';

  return (
    <div
      className={`${styles.card} ${isHighlighted ? styles.highlighted : ''}`}
      onClick={handleClick}
      data-molecule-id={molecule.molecule_card_id}
    >
      <div className={styles.header}>
        <span className={styles.index}>#{index + 1}</span>
        <span className={styles.label}>Molecule</span>
        <span className={styles.statusBadge} data-status={molecule.normalization_status}>
          {molecule.normalization_status}
        </span>
        {molecule.source_images.length > 0 && (
          <span className={styles.pageBadge}>
            P{molecule.source_images[0].provenance.page}
          </span>
        )}
      </div>

      <div className={styles.body}>
        <div className={styles.structureArea}>
          <canvas ref={canvasRef} width={180} height={140} className={styles.structureCanvas} />
        </div>

        {figureUrl && sourceImage?.bbox && (
          <div className={styles.croppedArea}>
            <canvas ref={croppedRef} className={styles.croppedCanvas} />
          </div>
        )}
      </div>

      <div className={styles.info}>
        {displayName && (
          <div className={styles.nameRow}>
            <span className={styles.infoLabel}>Name:</span>
            <span className={styles.infoValue}>{displayName}</span>
          </div>
        )}
        <div className={styles.smilesRow}>
          <span className={styles.infoLabel}>SMILES:</span>
          <span className={styles.smilesValue} title={smiles}>
            {smiles.length > 60 ? smiles.slice(0, 60) + '...' : smiles}
          </span>
        </div>
        {molecule.inchi_key && (
          <div className={styles.infoRow}>
            <span className={styles.infoLabel}>InChIKey:</span>
            <span className={styles.infoValue}>{molecule.inchi_key}</span>
          </div>
        )}
        {molecule.confidence > 0 && (
          <div className={styles.confRow}>
            <span className={styles.infoLabel}>Confidence:</span>
            <div className={styles.confBar}>
              <div
                className={styles.confFill}
                style={{ width: `${Math.min(100, molecule.confidence * 100)}%` }}
              />
            </div>
            <span className={styles.confText}>{Math.round(molecule.confidence * 100)}%</span>
          </div>
        )}
      </div>
    </div>
  );
}
