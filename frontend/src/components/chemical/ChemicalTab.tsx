import { useMemo } from 'react';
import MoleculeCard from './MoleculeCard';
import ReactionCard from './ReactionCard';
import type { DocumentData } from '../../hooks/useDocumentData';
import styles from './ChemicalTab.module.css';

interface Props {
  docId: string;
  docData: DocumentData;
}

export default function ChemicalTab({ docId, docData }: Props) {
  const molecules = docData.molecules;
  const reactions = docData.reactions;

  const molNamesMap = useMemo(() => {
    const map = new Map<string, typeof docData.molNames[0]>();
    for (const mn of docData.molNames) {
      map.set(mn.compound_label, mn);
    }
    return map;
  }, [docData.molNames]);

  if (molecules.length === 0 && reactions.length === 0) {
    return <div className={styles.empty}>No chemical elements extracted for this document.</div>;
  }

  return (
    <div className={styles.container}>
      {/* Molecules Section */}
      {molecules.length > 0 && (
        <div className={styles.section}>
          <div className={styles.sectionHeader}>
            <span className={styles.sectionTitle}>Molecules</span>
            <span className={styles.count}>{molecules.length}</span>
          </div>
          <div className={styles.cardList}>
            {molecules.map((mol, idx) => (
              <MoleculeCard
                key={mol.molecule_card_id}
                molecule={mol}
                docId={docId}
                index={idx}
                molName={molNamesMap.get(mol.molecule_card_id)}
              />
            ))}
          </div>
        </div>
      )}

      {/* Reactions Section */}
      {reactions.length > 0 && (
        <div className={styles.section}>
          <div className={styles.sectionHeader}>
            <span className={styles.sectionTitle}>Reactions</span>
            <span className={styles.count}>{reactions.length}</span>
          </div>
          <div className={styles.cardList}>
            {reactions.map((rxn, idx) => (
              <ReactionCard
                key={rxn.reaction_event_id}
                reaction={rxn}
                docId={docId}
                index={idx}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
