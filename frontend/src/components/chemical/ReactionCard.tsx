import { useRef, useEffect, useCallback } from 'react';
import { useAppStore } from '../../stores/useAppStore';
import type { ReactionEvent } from '../../types';
import styles from './ReactionCard.module.css';

interface Props {
  reaction: ReactionEvent;
  docId: string;
  index: number;
}

function StructureSmiles({ smiles }: { smiles: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!smiles || !canvasRef.current) return;
    if (smiles === '<invalid>' || smiles === '—') return;

    import('smiles-drawer').then(({ default: SmilesDrawer }) => {
      if (!canvasRef.current) return;
      try {
        const drawer = new SmilesDrawer.SmiDrawer({ width: 120, height: 90 });
        SmilesDrawer.parse(smiles, (tree: unknown) => {
          drawer.draw(tree as never, canvasRef.current!, 'light');
        }, () => {
          const ctx = canvasRef.current!.getContext('2d')!;
          ctx.fillStyle = '#f5f5f5';
          ctx.fillRect(0, 0, 120, 90);
          ctx.fillStyle = '#999';
          ctx.font = '10px sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText('Invalid', 60, 50);
        });
      } catch {
        // Fallback
      }
    });
  }, [smiles]);

  if (!smiles || smiles === '<invalid>') {
    return <div className={styles.invalidSmiles}>—</div>;
  }

  return <canvas ref={canvasRef} width={120} height={90} className={styles.structureCanvas} />;
}

export default function ReactionCard({ reaction, docId, index }: Props) {
  const highlightedReactionId = useAppStore(s => s.highlightedReactionId);
  const highlightReaction = useAppStore(s => s.highlightReaction);
  const setCurrentPage = useAppStore(s => s.setCurrentPage);
  const setActiveTab = useAppStore(s => s.setActiveTab);

  const isHighlighted = highlightedReactionId === reaction.reaction_event_id;

  const handleClick = useCallback(() => {
    highlightReaction(reaction.reaction_event_id);
    if (reaction.source?.page) {
      setCurrentPage(reaction.source.page - 1);
      setActiveTab('markdown');
    }
  }, [reaction, highlightReaction, setCurrentPage, setActiveTab]);

  const reactants = reaction.reactants.filter(r => r.smiles && r.smiles !== '<invalid>');
  const products = reaction.products.filter(r => r.smiles && r.smiles !== '<invalid>');

  return (
    <div
      className={`${styles.card} ${isHighlighted ? styles.highlighted : ''}`}
      onClick={handleClick}
      data-reaction-id={reaction.reaction_event_id}
    >
      <div className={styles.header}>
        <span className={styles.index}>#{index + 1}</span>
        <span className={styles.label}>Reaction</span>
        {reaction.source?.page && (
          <span className={styles.pageBadge}>P{reaction.source.page}</span>
        )}
        {reaction.yield_value && (
          <span className={styles.yieldBadge}>
            Yield: {reaction.yield_value.raw_text}{reaction.yield_value.unit}
          </span>
        )}
      </div>

      <div className={styles.reactionScheme}>
        <div className={styles.side}>
          {reactants.length > 0 ? (
            reactants.map((r, i) => (
              <div key={i} className={styles.molBox}>
                <StructureSmiles smiles={r.smiles} />
                {r.name && <span className={styles.molName}>{r.name}</span>}
              </div>
            ))
          ) : (
            <div className={styles.noData}>No reactants</div>
          )}
        </div>

        <div className={styles.arrow}>→</div>

        <div className={styles.side}>
          {products.length > 0 ? (
            products.map((p, i) => (
              <div key={i} className={styles.molBox}>
                <StructureSmiles smiles={p.smiles} />
                {p.name && <span className={styles.molName}>{p.name}</span>}
              </div>
            ))
          ) : (
            <div className={styles.noData}>No products</div>
          )}
        </div>
      </div>

      {(reaction.temperature || reaction.time || reaction.procedure_text) && (
        <div className={styles.conditions}>
          {reaction.temperature && (
            <span className={styles.conditionItem}>🌡 {reaction.temperature}</span>
          )}
          {reaction.time && (
            <span className={styles.conditionItem}>⏱ {reaction.time}</span>
          )}
          {reaction.procedure_text && (
            <span className={styles.procedureText}>
              {reaction.procedure_text.length > 120
                ? reaction.procedure_text.slice(0, 120) + '...'
                : reaction.procedure_text}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
