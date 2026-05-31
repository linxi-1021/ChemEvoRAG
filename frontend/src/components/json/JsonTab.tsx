import { useState, useMemo, useCallback } from 'react';
import { Light as SyntaxHighlighter } from 'react-syntax-highlighter';
import json from 'react-syntax-highlighter/dist/esm/languages/hljs/json';
import atomOneLight from 'react-syntax-highlighter/dist/esm/styles/hljs/atom-one-light';
import type { DocumentData } from '../../hooks/useDocumentData';
import styles from './JsonTab.module.css';

SyntaxHighlighter.registerLanguage('json', json);

type JsonSource = 'content_list' | 'blocks' | 'molecules' | 'reactions';

interface Props {
  docId: string;
  docData: DocumentData;
}

const SOURCES: { key: JsonSource; label: string }[] = [
  { key: 'content_list', label: 'Content List V2' },
  { key: 'blocks', label: 'Evidence Blocks' },
  { key: 'molecules', label: 'Molecules' },
  { key: 'reactions', label: 'Reactions' },
];

export default function JsonTab({ docId, docData }: Props) {
  const [activeSource, setActiveSource] = useState<JsonSource>('content_list');
  const [expanded, setExpanded] = useState(true);

  const jsonString = useMemo(() => {
    switch (activeSource) {
      case 'content_list':
        return docData.contentList ? JSON.stringify(docData.contentList, null, 2) : 'null';
      case 'blocks':
        return JSON.stringify(docData.evidenceBlocks, null, 2);
      case 'molecules':
        return JSON.stringify(docData.molecules, null, 2);
      case 'reactions':
        return JSON.stringify(docData.reactions, null, 2);
      default:
        return 'null';
    }
  }, [activeSource, docData]);

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(jsonString);
  }, [jsonString]);

  const handleDownload = useCallback(() => {
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${docId}.${activeSource}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [jsonString, docId, activeSource]);

  return (
    <div className={styles.container}>
      <div className={styles.toolbar}>
        <div className={styles.sourceTabs}>
          {SOURCES.map(s => (
            <button
              key={s.key}
              className={`${styles.sourceTab} ${activeSource === s.key ? styles.active : ''}`}
              onClick={() => setActiveSource(s.key)}
            >
              {s.label}
            </button>
          ))}
        </div>
        <div className={styles.actions}>
          <button className={styles.actionBtn} onClick={handleCopy} title="Copy JSON">
            📋 Copy
          </button>
          <button className={styles.actionBtn} onClick={handleDownload} title="Download JSON">
            ⬇ Download
          </button>
          <button
            className={styles.actionBtn}
            onClick={() => setExpanded(!expanded)}
            title={expanded ? 'Collapse all' : 'Expand all'}
          >
            {expanded ? '▼' : '▶'} {expanded ? 'Collapse' : 'Expand'}
          </button>
        </div>
      </div>

      <div className={styles.jsonView}>
        {expanded ? (
          <SyntaxHighlighter
            language="json"
            style={atomOneLight}
            customStyle={{
              margin: 0,
              padding: '12px',
              fontSize: '11px',
              lineHeight: '1.5',
              background: 'transparent',
              maxHeight: 'calc(100vh - 200px)',
            }}
            wrapLongLines
          >
            {jsonString.length > 500000
              ? jsonString.slice(0, 500000) + '\n... (truncated)'
              : jsonString}
          </SyntaxHighlighter>
        ) : (
          <pre className={styles.collapsed}>
            {jsonString.length > 1000
              ? jsonString.slice(0, 1000) + '\n...'
              : jsonString}
          </pre>
        )}
      </div>
    </div>
  );
}
