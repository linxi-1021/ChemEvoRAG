import { useAppStore } from '../../stores/useAppStore';
import { DOCUMENTS } from '../../api/manifest';
import styles from './DocumentList.module.css';

export default function DocumentList() {
  const selectedDocId = useAppStore(s => s.selectedDocId);
  const selectDocument = useAppStore(s => s.selectDocument);

  return (
    <div className={styles.list}>
      {DOCUMENTS.map(doc => (
        <button
          key={doc.docId}
          className={`${styles.item} ${selectedDocId === doc.docId ? styles.active : ''}`}
          onClick={() => selectDocument(doc.docId)}
        >
          <span className={styles.icon}>📄</span>
          <span className={styles.label}>{doc.label}</span>
          <span className={styles.id}>{doc.docId}</span>
        </button>
      ))}
    </div>
  );
}
